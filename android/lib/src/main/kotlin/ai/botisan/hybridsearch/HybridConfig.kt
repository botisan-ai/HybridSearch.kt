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
    internal fun hnswConfig(): HnswConfig = HnswConfig(
        maxConnections = hnswMaxConnections,
        maxElements = hnswMaxElements,
        maxLayers = hnswMaxLayers,
        efConstruction = hnswEfConstruction,
        dimension = embeddingDimension,
        distanceType = distanceType,
    )
}

public data class HybridSearchResult<T>(
    val docId: Long,
    val score: Float,
    val document: T,
)
