package ai.botisan.hybridsearch

import ai.botisan.hnsw.HnswIndex
import ai.botisan.tantivy.TantivyDocumentAdapter
import ai.botisan.tantivy.TantivyDocumentWriter
import ai.botisan.tantivy.TantivyFieldMap
import ai.botisan.tantivy.TantivyQuery
import ai.botisan.tantivy.TantivySchema
import ai.botisan.tantivy.TantivyValue
import ai.botisan.tantivy.TypedTantivyIndex
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Hybrid (BM25 + vector) index — Kotlin port of HybridSearch.swift's
 * `HybridIndex` actor. Documents live in Tantivy (authoritative); vectors live
 * in HNSW keyed by a minted `__doc_id`; text+vector results fuse via weighted
 * reciprocal-rank fusion.
 *
 * Every public operation — including compound ones like [index] (add + commit)
 * and delete-by-field (lookup + delete) — runs under one outer mutex on
 * [Dispatchers.IO], so no other caller can interleave between their steps.
 *
 * Crash-safe commit protocol: each commit dumps vectors under a fresh
 * `hnsw-g<generation>` basename, commits Tantivy, atomically publishes
 * metadata naming the generation (plus the committed Tantivy `docCount`), and
 * only then sweeps older generations. Published metadata therefore always
 * names a complete dump: a crash mid-cycle leaves only sweepable strays, and
 * a crash between Tantivy's commit and the publish is detected on [load] as
 * [HybridSearchException.TornCommit] (docCount cross-check plus a
 * `__doc_id >= nextDocId` probe) — the affected vectors are not
 * reconstructable, so that state fails rather than silently reusing ids.
 *
 * hnsw_rs cannot dump a physically empty graph, so an empty generation has
 * **no HNSW files** (`hasVectorGraph` records which state to expect); missing
 * or partial files — the tombstone sidecar included — under a populated
 * marker are [HybridSearchException.VectorStateCorrupt], never a silent
 * downgrade to text-only search. Stale files that cannot be deleted fail
 * loudly ([clear]/[commit] throw) rather than resurrecting on reopen.
 *
 * Deliberate deviation from the Swift port source: RRF ties are broken
 * deterministically (score desc, then docId asc) — the Swift implementation's
 * dictionary-order tie-break is nondeterministic.
 */
public class HybridIndex<T> private constructor(
    private val baseDir: File,
    private val tantivyIndex: TypedTantivyIndex<DocRecord<T>>,
    private var hnswIndex: HnswIndex,
    public val schema: TantivySchema,
    private val config: HybridIndexConfig,
    private var nextDocId: Long,
    private var hasVectorGraph: Boolean,
    private var generation: Long,
    public val primaryIdField: String,
    private val schemaFingerprint: String,
) : AutoCloseable {

    private val mutex = Mutex()

    @Volatile
    private var closed = false

    private val metadataFile = File(baseDir, METADATA_FILE_NAME)
    private val defaultTextFields = schema.defaultTextFieldNames

    /** The user's document plus the minted internal doc id, as stored in Tantivy. */
    internal class DocRecord<T>(val docId: Long, docProvider: () -> T) {
        /**
         * Decoding is deferred so id-only paths (delete-by-field) can address
         * a record even when the user decoder would fail on it.
         */
        val doc: T by lazy(docProvider)

        constructor(doc: T, docId: Long) : this(docId, { doc })
    }

    /**
     * Wraps the user adapter: appends `__doc_id` on encode (after checking the
     * adapter neither writes the reserved field nor forgets the primary id) and
     * reads it back on decode.
     */
    private class DocIdAdapter<T>(
        private val user: TantivyDocumentAdapter<T>,
        private val primaryIdField: String,
    ) : TantivyDocumentAdapter<DocRecord<T>> {
        override fun encode(value: DocRecord<T>, doc: TantivyDocumentWriter) {
            user.encode(value.doc, doc)
            if (doc.valueCount(DOC_ID_FIELD) > 0) throw HybridSearchException.ReservedField(DOC_ID_FIELD)
            val idCount = doc.valueCount(primaryIdField)
            if (idCount != 1) throw HybridSearchException.InvalidPrimaryIdValue(primaryIdField, idCount)
            doc.u64(DOC_ID_FIELD, value.docId)
        }

        override fun decode(fields: TantivyFieldMap): DocRecord<T> {
            val docId = fields.u64(DOC_ID_FIELD) ?: throw HybridSearchException.MissingDocId()
            return DocRecord(docId) { user.decode(fields) }
        }
    }

    public companion object {
        private const val DOC_ID_FIELD = "__doc_id"
        private const val METADATA_FILE_NAME = "hybrid.meta.json"

        /** Vector dumps are generation-numbered so a commit never overwrites the published generation in place. */
        private fun hnswBasename(generation: Long): String = "hnsw-g$generation"

        /** Creates a new index; throws [HybridSearchException.IndexAlreadyExists] if one exists at [directory]. */
        public suspend fun <T> create(
            directory: File,
            schema: TantivySchema,
            adapter: TantivyDocumentAdapter<T>,
            config: HybridIndexConfig = HybridIndexConfig(),
            primaryIdField: String? = null,
        ): HybridIndex<T> = withContext(Dispatchers.IO) {
            if (File(directory, METADATA_FILE_NAME).exists()) throw HybridSearchException.IndexAlreadyExists()
            requireUserSchema(schema)
            val primary = resolvePrimaryIdField(schema, primaryIdField)
            directory.mkdirs()

            val effectiveSchema = schema.extending { u64Field(DOC_ID_FIELD) }
            val recordAdapter = DocIdAdapter(adapter, primary)
            val index = HybridIndex(
                baseDir = directory,
                tantivyIndex = openTantivy(directory, effectiveSchema, recordAdapter),
                hnswIndex = HnswIndex.create(config.hnswConfig()),
                schema = schema,
                config = config,
                nextDocId = 0,
                hasVectorGraph = false,
                generation = 0,
                primaryIdField = primary,
                schemaFingerprint = schema.fingerprint(),
            )
            index.persistMetadataLocked()
            index
        }

        public suspend fun <T> create(
            path: String,
            schema: TantivySchema,
            adapter: TantivyDocumentAdapter<T>,
            config: HybridIndexConfig = HybridIndexConfig(),
            primaryIdField: String? = null,
        ): HybridIndex<T> = create(File(path), schema, adapter, config, primaryIdField)

        /** Loads an existing index; throws [HybridSearchException.MetadataMissing] when absent. */
        public suspend fun <T> load(
            directory: File,
            schema: TantivySchema,
            adapter: TantivyDocumentAdapter<T>,
            primaryIdField: String? = null,
        ): HybridIndex<T> = withContext(Dispatchers.IO) {
            val metadataFile = File(directory, METADATA_FILE_NAME)
            if (!metadataFile.exists()) throw HybridSearchException.MetadataMissing()
            requireUserSchema(schema)

            val metadata = HybridMetadataStore.load(metadataFile)
            if (metadata.version != HybridIndexMetadata.CURRENT_VERSION) {
                throw HybridSearchException.MetadataCorrupt("version ${metadata.version}")
            }
            if (metadata.nextDocId < 0) {
                throw HybridSearchException.MetadataCorrupt("negative nextDocId ${metadata.nextDocId}")
            }
            if (metadata.generation < 0) {
                throw HybridSearchException.MetadataCorrupt("negative generation ${metadata.generation}")
            }
            if (metadata.docCount < 0) {
                throw HybridSearchException.MetadataCorrupt("negative docCount ${metadata.docCount}")
            }

            val primary = primaryIdField?.also {
                if (it !in schema.idFieldNames) throw HybridSearchException.InvalidPrimaryIdField(it)
            } ?: metadata.primaryIdField.also {
                if (it !in schema.idFieldNames) throw HybridSearchException.InvalidPrimaryIdField(it)
            }

            val fingerprint = schema.fingerprint()
            if (fingerprint != metadata.schemaFingerprint) {
                throw HybridSearchException.MetadataCorrupt("schema fingerprint mismatch")
            }

            val config = try {
                HybridIndexConfig(
                    embeddingDimension = metadata.embeddingDimension,
                    hnswMaxConnections = metadata.maxConnections,
                    hnswMaxElements = metadata.maxElements,
                    hnswMaxLayers = metadata.maxLayers,
                    hnswEfConstruction = metadata.efConstruction,
                    distanceType = metadata.distanceType,
                )
            } catch (e: IllegalArgumentException) {
                throw HybridSearchException.MetadataCorrupt(e.message ?: "invalid hnsw config")
            }

            val effectiveSchema = schema.extending { u64Field(DOC_ID_FIELD) }
            val recordAdapter = DocIdAdapter(adapter, primary)
            val tantivy = openTantivy(directory, effectiveSchema, recordAdapter)
            try {
                // Cross-check the committed Tantivy state before trusting the
                // metadata (or touching any vector file): a crash between
                // Tantivy's commit and the metadata publish leaves them
                // disagreeing, and the affected vectors cannot be
                // reconstructed, so that state must fail — not sweep the graph
                // as a stray or remint already-committed doc ids.
                val committedCount = tantivy.count()
                if (committedCount != metadata.docCount) {
                    throw HybridSearchException.TornCommit(
                        "Tantivy holds $committedCount committed documents but metadata recorded ${metadata.docCount}",
                    )
                }
                val probe = tantivy.search(
                    TantivyQuery.Range(DOC_ID_FIELD, lower = TantivyValue.U64(metadata.nextDocId)),
                    limit = 1,
                )
                if (probe.count > 0) {
                    throw HybridSearchException.TornCommit(
                        "Tantivy holds a document with $DOC_ID_FIELD >= recorded nextDocId ${metadata.nextDocId}",
                    )
                }

                // The committed metadata says which generation's HNSW files
                // must exist. A populated marker with any of the three files
                // missing (tombstone sidecar included — without it deleted
                // vectors would resurrect) is corruption, never silently an
                // empty or partial vector leg.
                val basename = hnswBasename(metadata.generation)
                val hnsw = if (metadata.hasVectorGraph) {
                    val missing = listOf("$basename.hnsw.graph", "$basename.hnsw.data", "$basename.deleted")
                        .filterNot { File(directory, it).exists() }
                    if (missing.isNotEmpty()) {
                        throw HybridSearchException.VectorStateCorrupt(
                            "metadata records vector generation ${metadata.generation} but ${missing.joinToString()} missing",
                        )
                    }
                    HnswIndex.load(
                        directory = directory,
                        basename = basename,
                        dimension = metadata.embeddingDimension,
                        distanceType = metadata.distanceType,
                        config = config.hnswConfig(),
                    )
                } else {
                    HnswIndex.create(config.hnswConfig())
                }
                // Only after every check passed are other hnsw* files provably
                // strays of interrupted or superseded cycles.
                try {
                    sweepHnswFiles(directory, keep = if (metadata.hasVectorGraph) basename else null)
                } catch (t: Throwable) {
                    hnsw.close()
                    throw t
                }
                HybridIndex(
                    baseDir = directory,
                    tantivyIndex = tantivy,
                    hnswIndex = hnsw,
                    schema = schema,
                    config = config,
                    nextDocId = metadata.nextDocId,
                    hasVectorGraph = metadata.hasVectorGraph,
                    generation = metadata.generation,
                    primaryIdField = primary,
                    schemaFingerprint = fingerprint,
                )
            } catch (t: Throwable) {
                tantivy.close()
                throw t
            }
        }

        public suspend fun <T> load(
            path: String,
            schema: TantivySchema,
            adapter: TantivyDocumentAdapter<T>,
            primaryIdField: String? = null,
        ): HybridIndex<T> = load(File(path), schema, adapter, primaryIdField)

        /** Loads when an index exists at [directory], otherwise creates one. */
        public suspend fun <T> open(
            directory: File,
            schema: TantivySchema,
            adapter: TantivyDocumentAdapter<T>,
            config: HybridIndexConfig = HybridIndexConfig(),
            primaryIdField: String? = null,
        ): HybridIndex<T> =
            if (File(directory, METADATA_FILE_NAME).exists()) {
                load(directory, schema, adapter, primaryIdField)
            } else {
                create(directory, schema, adapter, config, primaryIdField)
            }

        public suspend fun <T> open(
            path: String,
            schema: TantivySchema,
            adapter: TantivyDocumentAdapter<T>,
            config: HybridIndexConfig = HybridIndexConfig(),
            primaryIdField: String? = null,
        ): HybridIndex<T> = open(File(path), schema, adapter, config, primaryIdField)

        private fun requireUserSchema(schema: TantivySchema) {
            if (DOC_ID_FIELD in schema.fieldNames) throw HybridSearchException.ReservedField(DOC_ID_FIELD)
        }

        private fun resolvePrimaryIdField(schema: TantivySchema, requested: String?): String {
            val idFields = schema.idFieldNames
            if (idFields.isEmpty()) throw HybridSearchException.MissingIdField()
            return when {
                requested != null ->
                    if (requested in idFields) requested else throw HybridSearchException.InvalidPrimaryIdField(requested)
                idFields.size == 1 -> idFields[0]
                else -> throw HybridSearchException.AmbiguousIdField(idFields)
            }
        }

        private suspend fun <T> openTantivy(
            baseDir: File,
            effectiveSchema: TantivySchema,
            recordAdapter: DocIdAdapter<T>,
        ): TypedTantivyIndex<DocRecord<T>> {
            val tantivyDir = File(baseDir, "tantivy")
            tantivyDir.mkdirs()
            return TypedTantivyIndex.open(tantivyDir, effectiveSchema, recordAdapter)
        }

        /** The exact pre-generational dump names this package once wrote. */
        private val LEGACY_HNSW_FILES = setOf("hnsw.hnsw.graph", "hnsw.hnsw.data", "hnsw.deleted")

        /** A generation-numbered dump/sidecar file, as written by [hnswBasename]. */
        private val GENERATION_HNSW_FILE = Regex("""^hnsw-g\d+\.(hnsw\.graph|hnsw\.data|deleted)$""")

        /**
         * Removes every HNSW dump/sidecar file this package recognizes as its
         * own — the exact legacy names and `hnsw-g<n>` generation triplets,
         * never arbitrary same-prefix files the caller keeps in the directory —
         * except the [keep] generation's (all of them when [keep] is null).
         * Throws when the directory cannot be enumerated or a stale file
         * survives deletion: stale dumps must not resurrect on a later load,
         * and a skipped sweep must not read as a clean one.
         */
        private fun sweepHnswFiles(directory: File, keep: String?) {
            val files = directory.listFiles()
                ?: throw HybridSearchException.VectorStateCorrupt(
                    "could not enumerate ${directory.absolutePath} for stale vector files",
                )
            val keepNames = keep?.let { setOf("$it.hnsw.graph", "$it.hnsw.data", "$it.deleted") }.orEmpty()
            for (file in files) {
                if (!file.isFile) continue
                val name = file.name
                if (name !in LEGACY_HNSW_FILES && !GENERATION_HNSW_FILE.matches(name)) continue
                if (name in keepNames) continue
                if (!file.delete()) {
                    throw HybridSearchException.VectorStateCorrupt(
                        "could not delete stale vector file ${file.absolutePath}",
                    )
                }
            }
        }
    }

    public val dimension: Int get() = config.embeddingDimension

    public suspend fun count(): Long = locked { tantivyIndex.count() }

    /**
     * Adds without committing; returns the minted docId. A failed call
     * publishes nothing: validation and the adapter encode reject before any
     * write, and a vector-leg failure masks the text write inside the same
     * open transaction (never an involuntary commit of other pending work).
     */
    public suspend fun add(doc: T, embedding: FloatArray): Long = locked { addLocked(doc, embedding) }

    public suspend fun addAll(docs: List<HybridDocument<T>>): List<Long> = locked { addAllLocked(docs) }

    /** Adds and commits — one atomic operation with respect to other callers. */
    public suspend fun index(doc: T, embedding: FloatArray): Long = locked {
        val docId = addLocked(doc, embedding)
        commitLocked()
        docId
    }

    public suspend fun indexAll(docs: List<HybridDocument<T>>): List<Long> = locked {
        val docIds = addAllLocked(docs)
        commitLocked()
        docIds
    }

    public suspend fun commit(): Unit = locked { commitLocked() }

    public suspend fun delete(docId: Long, persist: Boolean = true): Unit = locked {
        deleteLocked(docId, persist)
    }

    /** Deletes by an id field value; no-op when the document is absent (Swift parity). Lookup and delete run under one lock. */
    public suspend fun delete(idField: String, idValue: TantivyValue, persist: Boolean = true): Unit = locked {
        val record = tantivyIndex.getDocOrNull(idField, idValue) ?: return@locked
        deleteLocked(record.docId, persist)
    }

    public suspend fun get(docId: Long): T? = locked {
        tantivyIndex.getDocOrNull(DOC_ID_FIELD, TantivyValue.U64(docId))?.doc
    }

    public suspend fun get(idField: String, idValue: TantivyValue): T? = locked {
        tantivyIndex.getDocOrNull(idField, idValue)?.doc
    }

    public suspend fun searchText(
        query: HybridTextQuery,
        filter: TantivyQuery? = null,
        limit: Int = 10,
        offset: Int = 0,
    ): List<HybridSearchResult<T>> = locked {
        requirePaging(limit, offset)
        bm25SearchLocked(query, filter, limit, offset).map { (record, score) ->
            HybridSearchResult(record.docId, score, record.doc)
        }
    }

    public suspend fun searchText(
        query: String,
        filter: TantivyQuery? = null,
        limit: Int = 10,
        offset: Int = 0,
    ): List<HybridSearchResult<T>> = searchText(HybridTextQuery(query), filter, limit, offset)

    public suspend fun searchVector(
        embedding: FloatArray,
        filter: TantivyQuery? = null,
        limit: Int = 10,
        offset: Int = 0,
        efSearch: Int = 100,
        overfetchMultiplier: Int = 3,
    ): List<HybridSearchResult<T>> = locked {
        requirePaging(limit, offset)
        requireSearchControls(efSearch, overfetchMultiplier)
        val hits = vectorSearchHitsLocked(embedding, filter, limit, offset, efSearch, overfetchMultiplier)
        val page = hits.drop(offset).take(limit)
        val docsById = fetchDocsLocked(page.map { it.first })
        page.mapNotNull { (docId, distance) ->
            docsById[docId]?.let { HybridSearchResult(docId, similarity(distance), it) }
        }
    }

    public suspend fun searchHybrid(
        query: HybridTextQuery,
        embedding: FloatArray,
        filter: TantivyQuery? = null,
        limit: Int = 10,
        offset: Int = 0,
        efSearch: Int = 100,
        rrfK: Float = 60f,
        textWeight: Float = 1f,
        vectorWeight: Float = 1f,
        overfetchMultiplier: Int = 3,
    ): List<HybridSearchResult<T>> = locked {
        validateEmbedding(embedding)
        requirePaging(limit, offset)
        requireSearchControls(efSearch, overfetchMultiplier)
        require(rrfK.isFinite() && rrfK > 0f) { "rrfK must be finite and positive (got $rrfK)" }
        require(textWeight.isFinite() && textWeight >= 0f) { "textWeight must be finite and >= 0 (got $textWeight)" }
        require(vectorWeight.isFinite() && vectorWeight >= 0f) { "vectorWeight must be finite and >= 0 (got $vectorWeight)" }
        require(textWeight > 0f || vectorWeight > 0f) { "at least one of textWeight/vectorWeight must be positive" }

        val fetchLimit = fetchLimit(limit, offset, overfetchMultiplier)

        val bm25Hits = bm25SearchLocked(query, filter, fetchLimit, 0)
        val vectorHits = vectorSearchHitsLocked(embedding, filter, fetchLimit, 0, efSearch, overfetchMultiplier = 1)

        val ranked = rrfMerge(
            bm25 = bm25Hits.map { it.first.docId },
            vector = vectorHits.map { it.first },
            rrfK = rrfK,
            textWeight = textWeight,
            vectorWeight = vectorWeight,
        )

        val page = ranked.drop(offset).take(limit)
        val docsById = fetchDocsLocked(page.map { it.first })
        page.mapNotNull { (docId, score) ->
            docsById[docId]?.let { HybridSearchResult(docId, score, it) }
        }
    }

    public suspend fun searchHybrid(
        query: String,
        embedding: FloatArray,
        filter: TantivyQuery? = null,
        limit: Int = 10,
        offset: Int = 0,
        efSearch: Int = 100,
        rrfK: Float = 60f,
        textWeight: Float = 1f,
        vectorWeight: Float = 1f,
        overfetchMultiplier: Int = 3,
    ): List<HybridSearchResult<T>> = searchHybrid(
        HybridTextQuery(query),
        embedding,
        filter,
        limit,
        offset,
        efSearch,
        rrfK,
        textWeight,
        vectorWeight,
        overfetchMultiplier,
    )

    public suspend fun compact(): Unit = locked {
        hnswIndex.compact(config.hnswConfig())
        commitLocked() // compacting away the last points flips hasVectorGraph
    }

    /** Removes every document and vector, durably: reopening after clear yields an empty index. */
    public suspend fun clear(): Unit = locked {
        val newGeneration = mintableGeneration() // before Tantivy is durably cleared
        tantivyIndex.clear()
        // Files before state: if a stale dump cannot be removed, fail here —
        // before the metadata records an empty state a reopen would
        // contradict. The in-memory graph is still intact when that happens.
        sweepHnswFiles(baseDir, keep = null)
        hnswIndex.close()
        hnswIndex = HnswIndex.create(config.hnswConfig())
        nextDocId = 0
        generation = newGeneration
        hasVectorGraph = false
        persistMetadataLocked()
    }

    /** Waits for in-flight operations, then closes both underlying indexes. Idempotent. */
    override fun close() {
        runBlocking {
            mutex.withLock {
                if (!closed) {
                    closed = true
                    tantivyIndex.close()
                    hnswIndex.close()
                }
            }
        }
    }

    // -- locked internals (callers hold the mutex) -------------------------------

    private suspend fun addLocked(doc: T, embedding: FloatArray): Long {
        validateEmbedding(embedding)
        val docId = mintableDocId(count = 1)
        // Tantivy first: the write's own single adapter encode is the
        // validation (unknown/reserved fields, wrong value kinds, primary-id
        // cardinality), and it runs before the document reaches any FFI or
        // the insert-only vector graph. A rejected document therefore leaves
        // no tombstoned point and does not advance the id counter — and no
        // second encode exists for a stateful adapter to fail differently on.
        tantivyIndex.add(DocRecord(doc, docId))
        try {
            vectorInsertFailureForTest?.invoke(listOf(docId))
            hnswIndex.insert(embedding, docId)
        } catch (e: Exception) {
            rollBackDanglingText(listOf(docId), e)
            throw e
        }
        nextDocId = docId + 1
        return docId
    }

    private suspend fun addAllLocked(docs: List<HybridDocument<T>>): List<Long> {
        if (docs.isEmpty()) return emptyList()
        docs.forEach { validateEmbedding(it.embedding) }
        val first = mintableDocId(count = docs.size)
        val docIds = docs.indices.map { first + it }
        // Single encode per document, all before the batch reaches Tantivy's
        // FFI or the vector graph: one rejected document rejects the batch
        // with nothing mutated anywhere.
        tantivyIndex.addAll(docs.zip(docIds) { d, docId -> DocRecord(d.document, docId) })
        try {
            vectorInsertFailureForTest?.invoke(docIds)
            hnswIndex.insertBatch(docs.map { it.embedding }, docIds)
        } catch (e: Exception) {
            rollBackDanglingText(docIds, e)
            throw e
        }
        nextDocId = docIds.last() + 1
        return docIds
    }

    // Test seam: the only vector-leg failures reachable in production are
    // pathological (id collisions after external metadata damage, native
    // panics), so the rollback regressions inject one here — between the text
    // write and the HNSW insert — to stage the reviewer sequence
    // deterministically.
    internal var vectorInsertFailureForTest: ((List<Long>) -> Unit)? = null

    /**
     * A vector insert failed after its text write: stage deletes for the
     * uncommitted text documents in the same open transaction
     * ([TypedTantivyIndex.deleteDocWithoutCommit] masks exactly the documents
     * added before it), so this failed call publishes nothing — in particular
     * it cannot make another caller's still-uncommitted work durable the way
     * a committing delete would. The tombstone attempt is a no-op on every
     * prevalidated failure path (no point landed, so the graph never
     * registered the id); the ids are burned regardless, because a tombstoned
     * or unknown-partial id must never be reminted. Rollback failures ride
     * along as suppressed exceptions.
     */
    private suspend fun rollBackDanglingText(docIds: List<Long>, cause: Exception) {
        try {
            docIds.forEach { tantivyIndex.deleteDocWithoutCommit(DOC_ID_FIELD, TantivyValue.U64(it)) }
            hnswIndex.delete(docIds)
            nextDocId = docIds.last() + 1
        } catch (rollback: Exception) {
            cause.addSuppressed(rollback)
        }
    }

    /**
     * First id of a [count]-sized mint, verified against the Long domain
     * before anything mutates — the counter must never wrap.
     */
    private fun mintableDocId(count: Int): Long {
        check(nextDocId <= Long.MAX_VALUE - count) {
            "__doc_id space exhausted (nextDocId=$nextDocId, adding $count)"
        }
        return nextDocId
    }

    /**
     * The generation the next publish will use, verified against the Long
     * domain before anything mutates — like [mintableDocId], the counter must
     * never wrap: a wrapped negative generation would commit Tantivy and then
     * publish metadata every later [load] rejects as corrupt.
     */
    private fun mintableGeneration(): Long {
        check(generation != Long.MAX_VALUE) {
            "commit generation exhausted (generation=$generation)"
        }
        return generation + 1
    }

    /**
     * The publish protocol behind every durable state change: dump the new
     * vector generation, commit Tantivy, atomically publish metadata naming
     * both, then sweep older generations. A crash before the publish leaves
     * the previous generation intact (the new files are sweepable strays); a
     * crash after Tantivy's commit but before the publish is detected on load
     * as [HybridSearchException.TornCommit].
     */
    private suspend fun commitLocked() {
        val newGeneration = mintableGeneration()
        persistHnswLocked(newGeneration)
        tantivyIndex.commit()
        generation = newGeneration
        persistMetadataLocked()
        hnswIndex.setSearchingMode(true)
        sweepHnswFiles(baseDir, keep = if (hasVectorGraph) hnswBasename(newGeneration) else null)
    }

    private suspend fun deleteLocked(docId: Long, persist: Boolean) {
        tantivyIndex.deleteDoc(DOC_ID_FIELD, TantivyValue.U64(docId))
        hnswIndex.delete(docId)
        if (persist) {
            commitLocked()
        }
    }

    /**
     * Dumps vector state for [newGeneration] and tracks the marker the
     * metadata will publish. A physically empty graph cannot be dumped
     * (hnsw_rs limitation), so empty == no files for that generation; a
     * fully-tombstoned graph still dumps, carrying its tombstones. The
     * previous generation's files are untouched until the post-publish sweep.
     */
    private suspend fun persistHnswLocked(newGeneration: Long) {
        if (hnswIndex.graphSize() == 0L) {
            hasVectorGraph = false
        } else {
            hnswIndex.save(baseDir, hnswBasename(newGeneration))
            hasVectorGraph = true
        }
    }

    private fun validateEmbedding(embedding: FloatArray) {
        if (embedding.size != config.embeddingDimension) {
            throw HybridSearchException.DimensionMismatch(config.embeddingDimension, embedding.size)
        }
    }

    private fun requirePaging(limit: Int, offset: Int) {
        require(limit > 0) { "limit must be positive (got $limit)" }
        require(offset >= 0) { "offset must be non-negative (got $offset)" }
    }

    private fun requireSearchControls(efSearch: Int, overfetchMultiplier: Int) {
        require(efSearch > 0) { "efSearch must be positive (got $efSearch)" }
        require(overfetchMultiplier >= 1) { "overfetchMultiplier must be >= 1 (got $overfetchMultiplier)" }
    }

    private fun fetchLimit(limit: Int, offset: Int, overfetchMultiplier: Int): Int {
        val desired = limit.toLong() + offset.toLong()
        return (desired * overfetchMultiplier).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
    }

    private suspend fun persistMetadataLocked() {
        HybridMetadataStore.save(
            HybridIndexMetadata(
                version = HybridIndexMetadata.CURRENT_VERSION,
                embeddingDimension = config.embeddingDimension,
                distanceType = config.distanceType,
                maxConnections = config.hnswMaxConnections,
                maxElements = config.hnswMaxElements,
                maxLayers = config.hnswMaxLayers,
                efConstruction = config.hnswEfConstruction,
                nextDocId = nextDocId,
                primaryIdField = primaryIdField,
                schemaFingerprint = schemaFingerprint,
                hasVectorGraph = hasVectorGraph,
                generation = generation,
                docCount = tantivyIndex.count(),
            ),
            metadataFile,
        )
    }

    private suspend fun bm25SearchLocked(
        query: HybridTextQuery,
        filter: TantivyQuery?,
        limit: Int,
        offset: Int,
    ): List<Pair<DocRecord<T>, Float>> {
        val combined = combine(query.toTantivyQuery(defaultTextFields), filter)
        return tantivyIndex.search(combined, limit, offset).hits.map { it.doc to it.score }
    }

    private suspend fun vectorSearchHitsLocked(
        embedding: FloatArray,
        filter: TantivyQuery?,
        limit: Int,
        offset: Int,
        efSearch: Int,
        overfetchMultiplier: Int,
    ): List<Pair<Long, Float>> {
        validateEmbedding(embedding)
        val fetchLimit = fetchLimit(limit, offset, overfetchMultiplier)
        val effectiveEf = maxOf(efSearch, fetchLimit)

        hnswIndex.setSearchingMode(true)
        val results = hnswIndex.search(embedding, fetchLimit, effectiveEf).map { it.id to it.distance }
        if (filter == null) return results

        val allowed = filterDocIdsLocked(results.map { it.first }, filter)
        return results.filter { it.first in allowed }
    }

    private suspend fun filterDocIdsLocked(candidateIds: List<Long>, filter: TantivyQuery): Set<Long> {
        if (candidateIds.isEmpty()) return emptySet()
        val idQuery = TantivyQuery.TermSet(candidateIds.map { DOC_ID_FIELD to TantivyValue.U64(it) })
        val combined = TantivyQuery.Boolean(
            listOf(
                TantivyQuery.Clause(TantivyQuery.Occur.MUST, idQuery),
                TantivyQuery.Clause(TantivyQuery.Occur.MUST, filter),
            ),
        )
        return tantivyIndex.search(combined, limit = candidateIds.size)
            .hits.map { it.doc.docId }.toSet()
    }

    private suspend fun fetchDocsLocked(docIds: List<Long>): Map<Long, T> {
        if (docIds.isEmpty()) return emptyMap()
        return tantivyIndex.getDocs(docIds.map { DOC_ID_FIELD to TantivyValue.U64(it) })
            .associate { it.docId to it.doc }
    }

    private fun combine(query: TantivyQuery, filter: TantivyQuery?): TantivyQuery {
        if (filter == null) return query
        if (query is TantivyQuery.All) return filter
        return TantivyQuery.Boolean(
            listOf(
                TantivyQuery.Clause(TantivyQuery.Occur.MUST, query),
                TantivyQuery.Clause(TantivyQuery.Occur.MUST, filter),
            ),
        )
    }

    private fun rrfMerge(
        bm25: List<Long>,
        vector: List<Long>,
        rrfK: Float,
        textWeight: Float,
        vectorWeight: Float,
    ): List<Pair<Long, Float>> {
        val scores = mutableMapOf<Long, Float>()
        bm25.forEachIndexed { rank, docId ->
            scores.merge(docId, textWeight / (rrfK + (rank + 1)), Float::plus)
        }
        vector.forEachIndexed { rank, docId ->
            scores.merge(docId, vectorWeight / (rrfK + (rank + 1)), Float::plus)
        }
        return scores.entries
            .sortedWith(compareByDescending<Map.Entry<Long, Float>> { it.value }.thenBy { it.key })
            .map { it.key to it.value }
    }

    private fun similarity(distance: Float): Float = 1.0f / (1.0f + distance)

    private suspend fun <R> locked(block: suspend () -> R): R =
        withContext(Dispatchers.IO) {
            mutex.withLock {
                if (closed) throw HybridSearchException.AlreadyClosed()
                block()
            }
        }
}
