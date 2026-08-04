package ai.botisan.hybridsearch

import ai.botisan.hnsw.HnswConfig
import ai.botisan.hnsw.HnswDistanceType

public sealed class HybridSearchException(message: String) : Exception(message) {
    public class MetadataMissing : HybridSearchException("hybrid.meta.json not found — not a hybrid index directory")

    public class MetadataCorrupt(detail: String) : HybridSearchException("Hybrid metadata corrupt/incompatible: $detail")

    public class IndexAlreadyExists : HybridSearchException("A hybrid index already exists at this path")

    public class MissingIdField : HybridSearchException("Schema declares no idField — hybrid indexes require one")

    public class AmbiguousIdField(public val candidates: List<String>) :
        HybridSearchException("Multiple idFields declared (${candidates.joinToString()}); pass primaryIdField explicitly")

    public class InvalidPrimaryIdField(public val name: String) :
        HybridSearchException("primaryIdField '$name' is not an idField of the schema")

    public class DimensionMismatch(public val expected: Int, public val got: Int) :
        HybridSearchException("Embedding dimension mismatch: expected=$expected, got=$got")

    public class MissingDocId : HybridSearchException("Stored document has no __doc_id field")

    /** The schema or adapter used the reserved internal `__doc_id` field. */
    public class ReservedField(public val name: String) :
        HybridSearchException("'$name' is reserved for HybridIndex internals")

    /** Every document must carry exactly one value for the primary id field. */
    public class InvalidPrimaryIdValue(public val field: String, public val count: Int) :
        HybridSearchException("Document must carry exactly one '$field' value (got $count)")

    /**
     * The on-disk vector state contradicts the committed metadata (graph files
     * missing/partial for a populated index, or stale files that cannot be
     * removed). Nothing is silently downgraded to text-only search.
     */
    public class VectorStateCorrupt(detail: String) :
        HybridSearchException("Hybrid vector state corrupt: $detail")

    /**
     * The committed Tantivy state disagrees with the committed metadata: a
     * crash interrupted a commit cycle between Tantivy's own commit and the
     * metadata publish. Failing is deliberate — vectors for the affected
     * documents cannot be reconstructed, so the caller must rebuild or
     * restore the index rather than silently reuse document ids.
     */
    public class TornCommit(detail: String) :
        HybridSearchException("Hybrid commit was interrupted: $detail")

    public class AlreadyClosed : HybridSearchException("Index is closed")
}

/** Mirrors HybridSearch.swift's `HybridIndexConfig` defaults (384-d cosine). */
public data class HybridIndexConfig(
    val embeddingDimension: Int = 384,
    val hnswMaxConnections: Int = 16,
    val hnswMaxElements: Long = 100_000,
    val hnswMaxLayers: Int = 16,
    val hnswEfConstruction: Int = 200,
    val distanceType: HnswDistanceType = HnswDistanceType.COSINE,
) {
    init {
        // Same bounds HnswConfig enforces — checked here so an invalid config
        // fails at construction, not at first use.
        require(embeddingDimension >= 1) { "embeddingDimension must be >= 1 (got $embeddingDimension)" }
        require(hnswMaxConnections in 2..255) { "hnswMaxConnections must be in 2..255 (got $hnswMaxConnections)" }
        require(hnswMaxElements in 1..HnswConfig.MAX_ELEMENTS_LIMIT) {
            "hnswMaxElements must be in 1..${HnswConfig.MAX_ELEMENTS_LIMIT} (got $hnswMaxElements)"
        }
        require(hnswMaxLayers in 1..16) { "hnswMaxLayers must be in 1..16 (got $hnswMaxLayers)" }
        require(hnswEfConstruction >= 1) { "hnswEfConstruction must be >= 1 (got $hnswEfConstruction)" }
    }

    internal fun hnswConfig(): HnswConfig = HnswConfig(
        maxConnections = hnswMaxConnections,
        maxElements = hnswMaxElements,
        maxLayers = hnswMaxLayers,
        efConstruction = hnswEfConstruction,
        dimension = embeddingDimension,
        distanceType = distanceType,
    )
}

/** A document plus its embedding — the unit [HybridIndex.addAll]/[HybridIndex.indexAll] consume. */
public data class HybridDocument<T>(
    val document: T,
    val embedding: FloatArray,
) {
    override fun equals(other: Any?): Boolean =
        other is HybridDocument<*> && document == other.document && embedding.contentEquals(other.embedding)

    override fun hashCode(): Int = 31 * (document?.hashCode() ?: 0) + embedding.contentHashCode()
}

public data class HybridSearchResult<T>(
    val docId: Long,
    val score: Float,
    val document: T,
)
