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
 * Durable empty state: **missing HNSW files mean an empty vector graph.**
 * [commit] skips the vector dump while the graph is physically empty (hnsw_rs
 * cannot dump zero points) and deletes stale graph files; [clear] deletes
 * them; loading a directory without them starts a fresh in-memory graph. A
 * freshly created (or crashed-before-first-commit) index therefore reopens
 * fine.
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
    public val primaryIdField: String,
    private val schemaFingerprint: String,
) : AutoCloseable {

    private val mutex = Mutex()

    @Volatile
    private var closed = false

    private val metadataFile = File(baseDir, METADATA_FILE_NAME)
    private val defaultTextFields = schema.defaultTextFieldNames

    /** The user's document plus the minted internal doc id, as stored in Tantivy. */
    internal class DocRecord<T>(val doc: T, val docId: Long)

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
            return DocRecord(user.decode(fields), docId)
        }
    }

    public companion object {
        private const val DOC_ID_FIELD = "__doc_id"
        private const val METADATA_FILE_NAME = "hybrid.meta.json"
        private const val HNSW_BASENAME = "hnsw"

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

            val index = HybridIndex(
                baseDir = directory,
                tantivyIndex = openTantivy(directory, schema, adapter, primary),
                hnswIndex = HnswIndex(config.hnswConfig()),
                schema = schema,
                config = config,
                nextDocId = 0,
                primaryIdField = primary,
                schemaFingerprint = schema.fingerprint(),
            )
            index.persistMetadata()
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

            // Missing graph files == empty vector graph (fresh create, clear(),
            // or a crash before the first non-empty commit).
            val hnsw = if (File(directory, "$HNSW_BASENAME.hnsw.graph").exists()) {
                HnswIndex.load(
                    directory = directory,
                    basename = HNSW_BASENAME,
                    dimension = metadata.embeddingDimension,
                    distanceType = metadata.distanceType,
                    config = config.hnswConfig(),
                )
            } else {
                File(directory, "$HNSW_BASENAME.deleted").delete() // stray sidecar without a graph
                HnswIndex(config.hnswConfig())
            }

            HybridIndex(
                baseDir = directory,
                tantivyIndex = openTantivy(directory, schema, adapter, primary),
                hnswIndex = hnsw,
                schema = schema,
                config = config,
                nextDocId = metadata.nextDocId,
                primaryIdField = primary,
                schemaFingerprint = fingerprint,
            )
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
            schema: TantivySchema,
            adapter: TantivyDocumentAdapter<T>,
            primaryIdField: String,
        ): TypedTantivyIndex<DocRecord<T>> {
            val effectiveSchema = schema.extending { u64Field(DOC_ID_FIELD) }
            val tantivyDir = File(baseDir, "tantivy")
            tantivyDir.mkdirs()
            return TypedTantivyIndex.open(tantivyDir, effectiveSchema, DocIdAdapter(adapter, primaryIdField))
        }
    }

    public val dimension: Int get() = config.embeddingDimension

    public suspend fun count(): Long = locked { tantivyIndex.count() }

    /** Adds without committing; returns the minted docId. Rolls back the HNSW insert if Tantivy fails. */
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
        persistHnswLocked()
    }

    /** Removes every document and vector, durably: reopening after clear yields an empty index. */
    public suspend fun clear(): Unit = locked {
        tantivyIndex.clear()
        hnswIndex.close()
        deleteHnswFiles()
        hnswIndex = HnswIndex(config.hnswConfig())
        nextDocId = 0
        persistMetadata()
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
        val docId = nextDocId
        nextDocId += 1

        hnswIndex.insert(embedding, docId)
        try {
            tantivyIndex.add(DocRecord(doc, docId))
        } catch (e: Exception) {
            hnswIndex.delete(docId)
            throw e
        }
        return docId
    }

    private suspend fun addAllLocked(docs: List<HybridDocument<T>>): List<Long> {
        if (docs.isEmpty()) return emptyList()
        docs.forEach { validateEmbedding(it.embedding) }

        val docIds = docs.indices.map { nextDocId + it }
        nextDocId += docs.size

        hnswIndex.insertBatch(docs.map { it.embedding }, docIds)
        try {
            tantivyIndex.addAll(docs.zip(docIds) { d, docId -> DocRecord(d.document, docId) })
        } catch (e: Exception) {
            docIds.forEach { hnswIndex.delete(it) }
            throw e
        }
        return docIds
    }

    private suspend fun commitLocked() {
        tantivyIndex.commit()
        persistHnswLocked()
        hnswIndex.setSearchingMode(true)
        persistMetadata()
    }

    private suspend fun deleteLocked(docId: Long, persist: Boolean) {
        tantivyIndex.deleteDoc(DOC_ID_FIELD, TantivyValue.U64(docId))
        hnswIndex.delete(docId)
        if (persist) {
            persistHnswLocked()
            persistMetadata()
        }
    }

    /**
     * Persists vector state. A physically empty graph cannot be dumped
     * (hnsw_rs limitation), so empty == no files on disk; a fully-tombstoned
     * graph still dumps, carrying its tombstones.
     */
    private suspend fun persistHnswLocked() {
        if (hnswIndex.graphSize() == 0L) {
            deleteHnswFiles()
        } else {
            hnswIndex.save(baseDir, HNSW_BASENAME)
        }
    }

    private fun deleteHnswFiles() {
        File(baseDir, "$HNSW_BASENAME.hnsw.graph").delete()
        File(baseDir, "$HNSW_BASENAME.hnsw.data").delete()
        File(baseDir, "$HNSW_BASENAME.deleted").delete()
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

    private fun persistMetadata() {
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
