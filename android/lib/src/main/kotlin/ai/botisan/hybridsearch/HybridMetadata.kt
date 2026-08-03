package ai.botisan.hybridsearch

import ai.botisan.hnsw.HnswDistanceType
import java.io.File
import java.io.FileOutputStream
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long
import kotlinx.serialization.json.put

/**
 * `hybrid.meta.json` — same shape as HybridSearch.swift's metadata (version 1,
 * camelCase keys, distance types as "l2"/"cosine"/"dot"/"l1") plus the
 * Kotlin-only `hasVectorGraph` marker: the committed record of whether HNSW
 * files must exist on disk, so a missing graph is distinguishable corruption
 * rather than silently an empty index. The schemaFingerprint format also
 * differs (derived from the Kotlin schema DSL, not Swift reflection), so index
 * directories were never cross-language portable.
 */
internal data class HybridIndexMetadata(
    val version: Int,
    val embeddingDimension: Int,
    val distanceType: HnswDistanceType,
    val maxConnections: Int,
    val maxElements: Long,
    val maxLayers: Int,
    val efConstruction: Int,
    val nextDocId: Long,
    val primaryIdField: String,
    val schemaFingerprint: String,
    val hasVectorGraph: Boolean,
) {
    companion object {
        const val CURRENT_VERSION = 1
    }
}

internal object HybridMetadataStore {
    private val json = Json { prettyPrint = false }

    private fun distanceToString(d: HnswDistanceType): String = when (d) {
        HnswDistanceType.L2 -> "l2"
        HnswDistanceType.COSINE -> "cosine"
        HnswDistanceType.DOT -> "dot"
        HnswDistanceType.L1 -> "l1"
    }

    private fun distanceFromString(s: String): HnswDistanceType = when (s) {
        "l2" -> HnswDistanceType.L2
        "cosine" -> HnswDistanceType.COSINE
        "dot" -> HnswDistanceType.DOT
        "l1" -> HnswDistanceType.L1
        else -> throw HybridSearchException.MetadataCorrupt("unknown distanceType '$s'")
    }

    fun save(metadata: HybridIndexMetadata, file: File) {
        val obj = buildJsonObject {
            put("version", metadata.version)
            put("embeddingDimension", metadata.embeddingDimension)
            put("distanceType", distanceToString(metadata.distanceType))
            put(
                "hnswConfig",
                buildJsonObject {
                    put("maxConnections", metadata.maxConnections)
                    put("maxElements", metadata.maxElements)
                    put("maxLayers", metadata.maxLayers)
                    put("efConstruction", metadata.efConstruction)
                },
            )
            put("nextDocId", metadata.nextDocId)
            put("primaryIdField", metadata.primaryIdField)
            put("schemaFingerprint", metadata.schemaFingerprint)
            put("hasVectorGraph", metadata.hasVectorGraph)
        }
        val tmp = File(file.parentFile, "${file.name}.tmp")
        FileOutputStream(tmp).use { out ->
            out.write(obj.toString().toByteArray(Charsets.UTF_8))
            out.fd.sync()
        }
        // POSIX rename(2) atomically replaces within a directory and is
        // available on API 24 (java.nio.file needs API 26).
        if (!tmp.renameTo(file)) {
            tmp.delete()
            throw HybridSearchException.MetadataCorrupt("could not atomically replace ${file.absolutePath}")
        }
    }

    fun load(file: File): HybridIndexMetadata {
        val root = try {
            json.parseToJsonElement(file.readText()).jsonObject
        } catch (e: Exception) {
            throw HybridSearchException.MetadataCorrupt(e.message ?: "unparseable JSON")
        }
        try {
            val hnsw = root.getValue("hnswConfig").jsonObject
            return HybridIndexMetadata(
                version = root.getValue("version").jsonPrimitive.int,
                embeddingDimension = root.getValue("embeddingDimension").jsonPrimitive.int,
                distanceType = distanceFromString(root.getValue("distanceType").jsonPrimitive.content),
                maxConnections = hnsw.getValue("maxConnections").jsonPrimitive.int,
                maxElements = hnsw.getValue("maxElements").jsonPrimitive.long,
                maxLayers = hnsw.getValue("maxLayers").jsonPrimitive.int,
                efConstruction = hnsw.getValue("efConstruction").jsonPrimitive.int,
                nextDocId = root.getValue("nextDocId").jsonPrimitive.long,
                primaryIdField = root.getValue("primaryIdField").jsonPrimitive.content,
                schemaFingerprint = root.getValue("schemaFingerprint").jsonPrimitive.content,
                hasVectorGraph = root.getValue("hasVectorGraph").jsonPrimitive.boolean,
            )
        } catch (e: HybridSearchException) {
            throw e
        } catch (e: Exception) {
            throw HybridSearchException.MetadataCorrupt(e.message ?: "missing field")
        }
    }
}
