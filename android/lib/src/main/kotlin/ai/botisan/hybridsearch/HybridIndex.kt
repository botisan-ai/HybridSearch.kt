package ai.botisan.hybridsearch

import ai.botisan.hnsw.HnswIndex
import ai.botisan.tantivy.TantivyDocumentAdapter
import ai.botisan.tantivy.TantivyDocumentWriter
import ai.botisan.tantivy.TantivyFieldMap
import ai.botisan.tantivy.TantivyQuery
import ai.botisan.tantivy.TantivySchema
import ai.botisan.tantivy.TantivyValue
import ai.botisan.tantivy.ffi.DocumentField
import ai.botisan.tantivy.ffi.FieldValue
import ai.botisan.tantivy.ffi.NumericFieldOptions
import ai.botisan.tantivy.ffi.TantivyDocumentFields
import ai.botisan.tantivy.ffi.TantivyIndex
import ai.botisan.tantivy.ffi.TantivyIndexException
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/**
 * Hybrid (BM25 + vector) index — Kotlin port of HybridSearch.swift's
 * `HybridIndex` actor. Documents live in Tantivy (authoritative); vectors live
 * in HNSW keyed by a minted `__doc_id`; text+vector results fuse via weighted
 * reciprocal-rank fusion.
 *
 * Deliberate deviation from the Swift port source: RRF ties are broken
 * deterministically (score desc, then docId asc) — the Swift implementation's
 * dictionary-order tie-break is nondeterministic.
 */
public class HybridIndex<T> private constructor(
    private val baseDir: File,
    private val tantivyIndex: TantivyIndex,
    private var hnswIndex: HnswIndex,
    public val schema: TantivySchema,
    private val adapter: TantivyDocumentAdapter<T>,
    private val config: HybridIndexConfig,
    private var nextDocId: Long,
    public val primaryIdField: String,
    private val schemaFingerprint: String,
) : AutoCloseable {

    private val mutex = Mutex()
    private val metadataFile = File(baseDir, METADATA_FILE_NAME)
    private val defaultTextFields = schema.defaultTextFieldNames

    public companion object {
        private const val DOC_ID_FIELD = "__doc_id"
        private const val METADATA_FILE_NAME = "hybrid.meta.json"
        private const val HNSW_BASENAME = "hnsw"

        /** Creates a new index; throws [HybridSearchException.IndexAlreadyExists] if one exists at [path]. */
        public suspend fun <T> create(
            path: String,
            schema: TantivySchema,
            adapter: TantivyDocumentAdapter<T>,
            config: HybridIndexConfig = HybridIndexConfig(),
            primaryIdField: String? = null,
        ): HybridIndex<T> = withContext(Dispatchers.IO) {
            val baseDir = File(path)
            if (File(baseDir, METADATA_FILE_NAME).exists()) throw HybridSearchException.IndexAlreadyExists()
            val primary = resolvePrimaryIdField(schema, primaryIdField)
            baseDir.mkdirs()

            val tantivy = openTantivy(baseDir, schema)
            val hnsw = HnswIndex(config.hnswConfig())
            val index = HybridIndex(
                baseDir = baseDir,
                tantivyIndex = tantivy,
                hnswIndex = hnsw,
                schema = schema,
                adapter = adapter,
                config = config,
                nextDocId = 0,
                primaryIdField = primary,
                schemaFingerprint = schema.fingerprint(),
            )
            index.persistMetadata()
            index
        }

        /** Loads an existing index; throws [HybridSearchException.MetadataMissing] when absent. */
        public suspend fun <T> load(
            path: String,
            schema: TantivySchema,
            adapter: TantivyDocumentAdapter<T>,
            primaryIdField: String? = null,
        ): HybridIndex<T> = withContext(Dispatchers.IO) {
            val baseDir = File(path)
            val metadataFile = File(baseDir, METADATA_FILE_NAME)
            if (!metadataFile.exists()) throw HybridSearchException.MetadataMissing()

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

            val config = HybridIndexConfig(
                embeddingDimension = metadata.embeddingDimension,
                hnswMaxConnections = metadata.maxConnections,
                hnswMaxElements = metadata.maxElements,
                hnswMaxLayers = metadata.maxLayers,
                hnswEfConstruction = metadata.efConstruction,
                distanceType = metadata.distanceType,
            )
            val tantivy = openTantivy(baseDir, schema)
            val hnsw = HnswIndex.load(
                directory = baseDir.absolutePath,
                basename = HNSW_BASENAME,
                dimension = metadata.embeddingDimension,
                distanceType = metadata.distanceType,
                config = config.hnswConfig(),
            )

            HybridIndex(
                baseDir = baseDir,
                tantivyIndex = tantivy,
                hnswIndex = hnsw,
                schema = schema,
                adapter = adapter,
                config = config,
                nextDocId = metadata.nextDocId,
                primaryIdField = primary,
                schemaFingerprint = fingerprint,
            )
        }

        /** Loads when an index exists at [path], otherwise creates one. */
        public suspend fun <T> open(
            path: String,
            schema: TantivySchema,
            adapter: TantivyDocumentAdapter<T>,
            config: HybridIndexConfig = HybridIndexConfig(),
            primaryIdField: String? = null,
        ): HybridIndex<T> =
            if (File(path, METADATA_FILE_NAME).exists()) {
                load(path, schema, adapter, primaryIdField)
            } else {
                create(path, schema, adapter, config, primaryIdField)
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

        private fun openTantivy(baseDir: File, schema: TantivySchema): TantivyIndex {
            val builder = schema.newBuilder()
            builder.addU64Field(
                DOC_ID_FIELD,
                NumericFieldOptions(indexed = true, stored = true, fast = true, fieldnorms = false),
            )
            val tantivyDir = File(baseDir, "tantivy")
            tantivyDir.mkdirs()
            return TantivyIndex.newWithSchema(tantivyDir.absolutePath, builder)
        }
    }

    public fun dimension(): Int = config.embeddingDimension

    public suspend fun count(): Long = locked { tantivyIndex.docsCount().toLong() }

    /** Adds without committing; returns the minted docId. Rolls back the HNSW insert if Tantivy fails. */
    public suspend fun add(doc: T, embedding: FloatArray): Long = locked {
        validateEmbedding(embedding)
        val docId = nextDocId
        nextDocId += 1

        hnswInsert(embedding, docId)
        try {
            tantivyIndex.indexDoc(encodeWithDocId(doc, docId))
        } catch (e: Exception) {
            hnswDelete(docId)
            throw e
        }
        docId
    }

    public suspend fun addAll(docs: List<Pair<T, FloatArray>>): List<Long> = locked {
        if (docs.isEmpty()) return@locked emptyList()
        docs.forEach { validateEmbedding(it.second) }

        val docIds = docs.indices.map { nextDocId + it }
        nextDocId += docs.size

        val nativeDocs = docs.zip(docIds).map { (pair, docId) -> encodeWithDocId(pair.first, docId) }
        hnswInsertBatch(docs.map { it.second }, docIds)
        try {
            tantivyIndex.indexDocs(nativeDocs)
        } catch (e: Exception) {
            docIds.forEach { hnswDelete(it) }
            throw e
        }
        docIds
    }

    /** Adds and commits. */
    public suspend fun index(doc: T, embedding: FloatArray): Long {
        val docId = add(doc, embedding)
        commit()
        return docId
    }

    public suspend fun indexAll(docs: List<Pair<T, FloatArray>>): List<Long> {
        val docIds = addAll(docs)
        commit()
        return docIds
    }

    public suspend fun commit(): Unit = locked {
        tantivyIndex.commit()
        hnswSaveAndEnableSearch()
        persistMetadata()
    }

    public suspend fun delete(docId: Long, persist: Boolean = true): Unit = locked {
        tantivyIndex.deleteDoc(DocumentField(DOC_ID_FIELD, FieldValue.U64(docId.toULong())))
        hnswDelete(docId)
        if (persist) {
            hnswSave()
            persistMetadata()
        }
    }

    /** Deletes by an id field value; no-op when the document is absent (Swift parity). */
    public suspend fun delete(idField: String, idValue: TantivyValue, persist: Boolean = true) {
        val docId = locked {
            val fields = getDocFieldsOrNull(idField, idValue) ?: return@locked null
            docIdOf(fields) ?: throw HybridSearchException.MissingDocId()
        } ?: return
        delete(docId, persist)
    }

    public suspend fun get(docId: Long): T? = locked {
        getDocFieldsOrNull(DOC_ID_FIELD, TantivyValue.U64(docId))?.let { adapter.decode(TantivyFieldMap(it)) }
    }

    public suspend fun get(idField: String, idValue: TantivyValue): T? = locked {
        getDocFieldsOrNull(idField, idValue)?.let { adapter.decode(TantivyFieldMap(it)) }
    }

    public suspend fun searchText(
        query: HybridTextQuery,
        filter: TantivyQuery? = null,
        limit: Int = 10,
        offset: Int = 0,
    ): List<HybridSearchResult<T>> = locked {
        bm25Search(query, filter, limit, offset).mapNotNull { (docId, score, fields) ->
            HybridSearchResult(docId, score, adapter.decode(TantivyFieldMap(fields)))
        }
    }

    public suspend fun searchVector(
        embedding: FloatArray,
        filter: TantivyQuery? = null,
        limit: Int = 10,
        offset: Int = 0,
        efSearch: Int = 100,
        overfetchMultiplier: Int = 3,
    ): List<HybridSearchResult<T>> = locked {
        val hits = vectorSearchHits(embedding, filter, limit, offset, efSearch, overfetchMultiplier)
        val page = hits.drop(offset).take(limit)
        val docsById = fetchDocs(page.map { it.first })
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

        val desired = maxOf(1, limit + offset)
        val fetchLimit = maxOf(1, desired * overfetchMultiplier)

        val bm25Hits = bm25Search(query, filter, fetchLimit, 0)
        val vectorHits = vectorSearchHits(embedding, filter, fetchLimit, 0, efSearch, overfetchMultiplier = 1)

        val ranked = rrfMerge(
            bm25 = bm25Hits.map { it.first },
            vector = vectorHits.map { it.first },
            rrfK = rrfK,
            textWeight = textWeight,
            vectorWeight = vectorWeight,
        )

        val page = ranked.drop(offset).take(limit)
        val docsById = fetchDocs(page.map { it.first })
        page.mapNotNull { (docId, score) ->
            docsById[docId]?.let { HybridSearchResult(docId, score, it) }
        }
    }

    public suspend fun compact(): Unit = locked {
        hnswCompact()
        hnswSave()
    }

    public suspend fun clear(): Unit = locked {
        tantivyIndex.clearIndex()
        hnswIndex.close()
        hnswIndex = HnswIndex(config.hnswConfig())
        nextDocId = 0
        persistMetadata()
    }

    override fun close() {
        tantivyIndex.destroy()
        hnswIndex.close()
    }

    // -- internals -------------------------------------------------------------

    private fun validateEmbedding(embedding: FloatArray) {
        if (embedding.size != config.embeddingDimension) {
            throw HybridSearchException.DimensionMismatch(config.embeddingDimension, embedding.size)
        }
    }

    private fun encodeWithDocId(doc: T, docId: Long): TantivyDocumentFields {
        val writer = TantivyDocumentWriter()
        adapter.encode(doc, writer)
        writer.u64(DOC_ID_FIELD, docId)
        return writer.build()
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

    private fun getDocFieldsOrNull(field: String, value: TantivyValue): TantivyDocumentFields? = try {
        tantivyIndex.getDoc(DocumentField(field, value.toFfiValue()))
    } catch (_: TantivyIndexException.DocRetrievalException) {
        null
    }

    private fun bm25Search(
        query: HybridTextQuery,
        filter: TantivyQuery?,
        limit: Int,
        offset: Int,
    ): List<Triple<Long, Float, TantivyDocumentFields>> {
        val combined = combine(query.toTantivyQuery(defaultTextFields), filter)
        val results = tantivyIndex.searchDsl(combined.toJson(), limit.toUInt(), offset.toUInt())
        return results.docs.mapNotNull { result ->
            docIdOf(result.doc)?.let { Triple(it, result.score, result.doc) }
        }
    }

    private suspend fun vectorSearchHits(
        embedding: FloatArray,
        filter: TantivyQuery?,
        limit: Int,
        offset: Int,
        efSearch: Int,
        overfetchMultiplier: Int,
    ): List<Pair<Long, Float>> {
        validateEmbedding(embedding)
        val desired = maxOf(1, limit + offset)
        val fetchLimit = maxOf(1, desired * overfetchMultiplier)
        val effectiveEf = maxOf(efSearch, fetchLimit)

        hnswIndex.setSearchingMode(true)
        val results = hnswSearch(embedding, fetchLimit, effectiveEf)
        if (filter == null) return results

        val allowed = filterDocIds(results.map { it.first }, filter)
        return results.filter { it.first in allowed }
    }

    private fun filterDocIds(candidateIds: List<Long>, filter: TantivyQuery): Set<Long> {
        if (candidateIds.isEmpty()) return emptySet()
        val idQuery = TantivyQuery.TermSet(candidateIds.map { DOC_ID_FIELD to TantivyValue.U64(it) })
        val combined = TantivyQuery.Boolean(
            listOf(
                TantivyQuery.Clause(TantivyQuery.Occur.MUST, idQuery),
                TantivyQuery.Clause(TantivyQuery.Occur.MUST, filter),
            ),
        )
        val results = tantivyIndex.searchDsl(combined.toJson(), candidateIds.size.toUInt(), 0u)
        return results.docs.mapNotNull { docIdOf(it.doc) }.toSet()
    }

    private fun fetchDocs(docIds: List<Long>): Map<Long, T> {
        if (docIds.isEmpty()) return emptyMap()
        val fields = docIds.map { DocumentField(DOC_ID_FIELD, FieldValue.U64(it.toULong())) }
        return tantivyIndex.getDocsByIds(fields).mapNotNull { doc ->
            docIdOf(doc)?.let { it to adapter.decode(TantivyFieldMap(doc)) }
        }.toMap()
    }

    private fun docIdOf(fields: TantivyDocumentFields): Long? =
        TantivyFieldMap(fields).u64(DOC_ID_FIELD)

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
        withContext(Dispatchers.IO) { mutex.withLock { block() } }

    // HnswIndex's own API is suspend; these run inside our lock/IO context.
    private suspend fun hnswInsert(embedding: FloatArray, docId: Long) = hnswIndex.insert(embedding, docId)

    private suspend fun hnswInsertBatch(embeddings: List<FloatArray>, docIds: List<Long>) =
        hnswIndex.insertBatch(embeddings, docIds)

    private suspend fun hnswDelete(docId: Long) = hnswIndex.delete(docId)

    private suspend fun hnswSearch(embedding: FloatArray, k: Int, efSearch: Int) =
        hnswIndex.search(embedding, k, efSearch).map { it.id to it.distance }

    private suspend fun hnswSave() = hnswIndex.save(baseDir.absolutePath, HNSW_BASENAME)

    private suspend fun hnswSaveAndEnableSearch() {
        hnswSave()
        hnswIndex.setSearchingMode(true)
    }

    private suspend fun hnswCompact() = hnswIndex.compact(config.hnswConfig())
}

private fun TantivyValue.toFfiValue(): FieldValue = when (this) {
    is TantivyValue.Text -> FieldValue.Text(value)
    is TantivyValue.U64 -> FieldValue.U64(value.toULong())
    is TantivyValue.I64 -> FieldValue.I64(value)
    is TantivyValue.F64 -> FieldValue.F64(value)
    is TantivyValue.Bool -> FieldValue.Bool(value)
    is TantivyValue.DateMicros -> FieldValue.Date(epochMicros)
    is TantivyValue.Bytes -> FieldValue.Bytes(value)
    is TantivyValue.Facet -> FieldValue.Facet(path)
    is TantivyValue.Json -> FieldValue.Json(json)
}
