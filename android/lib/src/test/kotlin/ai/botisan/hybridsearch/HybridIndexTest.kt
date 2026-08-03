package ai.botisan.hybridsearch

import ai.botisan.tantivy.TantivyDocumentAdapter
import ai.botisan.tantivy.TantivyDocumentWriter
import ai.botisan.tantivy.TantivyFieldMap
import ai.botisan.tantivy.TantivyQuery
import ai.botisan.tantivy.TantivyValue
import ai.botisan.tantivy.tantivySchema
import java.nio.file.Files
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/** Port of HybridSearchSwiftTests (HybridSearch.swift @ 0.1.1) plus #195 round-trip additions. */
class HybridIndexTest {

    private data class TestDoc(
        val id: String,
        val title: String,
        val body: String,
        val isPublished: Boolean,
    )

    private val schema = tantivySchema {
        idField("id")
        textField("title")
        textField("body")
        boolField("isPublished")
    }

    private object TestDocAdapter : TantivyDocumentAdapter<TestDoc> {
        override fun encode(value: TestDoc, doc: TantivyDocumentWriter) {
            doc.text("id", value.id)
            doc.text("title", value.title)
            doc.text("body", value.body)
            doc.bool("isPublished", value.isPublished)
        }

        override fun decode(fields: TantivyFieldMap): TestDoc = TestDoc(
            id = fields.text("id")!!,
            title = fields.text("title")!!,
            body = fields.text("body")!!,
            isPublished = fields.bool("isPublished")!!,
        )
    }

    private val docs = listOf(
        TestDoc("swift-1", "Swift Concurrency", "Async await and actors in Swift.", true) to TestEmbeddings.docSwiftConcurrency,
        TestDoc("rust-1", "Rust FFI", "Calling Rust from Swift using UniFFI.", true) to TestEmbeddings.docRustFfi,
        TestDoc("vector-1", "Vector Search with HNSW", "Approximate nearest neighbor search using HNSW graphs.", false) to TestEmbeddings.docVectorHnsw,
        TestDoc("tantivy-1", "Full-text Search with Tantivy", "Indexing and BM25 ranking with Tantivy.", true) to TestEmbeddings.docTantivy,
    )

    private val config = HybridIndexConfig(embeddingDimension = 128, hnswMaxElements = 1_000)

    private fun tempPath(): String = Files.createTempDirectory("hybrid_test_").toFile().absolutePath

    private suspend fun makeSeededIndex(path: String = tempPath()): Pair<HybridIndex<TestDoc>, List<Long>> {
        val index = HybridIndex.create(path, schema, TestDocAdapter, config)
        val docIds = index.addAll(docs)
        index.commit()
        return index to docIds
    }

    @Test
    fun indexAndFetchById() = runTest {
        val (index, docIds) = makeSeededIndex()
        index.use {
            val fetched = it.get("id", TantivyValue.Text("swift-1"))
            assertEquals("Swift Concurrency", fetched?.title)

            val byDocId = it.get(docIds[0])
            assertEquals("swift-1", byDocId?.id)
        }
    }

    @Test
    fun textSearchReturnsSwiftResult() = runTest {
        val (index, _) = makeSeededIndex()
        index.use {
            val results = it.searchText(
                HybridTextQuery("swift actors", defaultFields = listOf("title", "body")),
                limit = 3,
            )
            assertEquals("swift-1", results.first().document.id)
        }
    }

    @Test
    fun vectorSearchAppliesFilters() = runTest {
        val (index, _) = makeSeededIndex()
        index.use {
            val filter = TantivyQuery.Term("isPublished", TantivyValue.Bool(true))
            val results = it.searchVector(
                embedding = TestEmbeddings.queryVector,
                filter = filter,
                limit = 3,
                efSearch = 50,
            )
            assertTrue(results.isNotEmpty())
            assertTrue(results.all { r -> r.document.isPublished })
            assertTrue(results.first().document.id != "vector-1")
        }
    }

    @Test
    fun hybridSearchUsesRrf() = runTest {
        val (index, _) = makeSeededIndex()
        index.use {
            val results = it.searchHybrid(
                query = HybridTextQuery("swift concurrency actors", defaultFields = listOf("title", "body")),
                embedding = TestEmbeddings.querySwift,
                limit = 3,
                efSearch = 50,
            )
            assertEquals("swift-1", results.first().document.id)
        }
    }

    @Test
    fun deleteRemovesDocuments() = runTest {
        val (index, _) = makeSeededIndex()
        index.use {
            it.delete("id", TantivyValue.Text("rust-1"))
            assertNull(it.get("id", TantivyValue.Text("rust-1")))

            val results = it.searchText(
                HybridTextQuery("Rust", defaultFields = listOf("title", "body")),
                limit = 5,
            )
            assertTrue(results.all { r -> r.document.id != "rust-1" })
        }
    }

    // -- additions beyond the Swift suite (#195 round-trip criteria) ------------

    @Test
    fun hybridFusesLexicalOnlyAndVectorOnlyHits() = runTest {
        val (index, _) = makeSeededIndex()
        index.use {
            // Text leg matches tantivy-1 lexically; vector leg's nearest doc is vector-1.
            val results = it.searchHybrid(
                query = HybridTextQuery("tantivy bm25 indexing", defaultFields = listOf("title", "body")),
                embedding = TestEmbeddings.queryVector,
                limit = 4,
                efSearch = 50,
            )
            val ids = results.map { r -> r.document.id }
            assertTrue("lexical-only hit missing from RRF fusion: $ids", "tantivy-1" in ids)
            assertTrue("vector-only hit missing from RRF fusion: $ids", "vector-1" in ids)
        }
    }

    @Test
    fun commitThenReopenKeepsSearchingAcrossLoad() = runTest {
        val path = tempPath()
        val (index, _) = makeSeededIndex(path)
        index.close()

        HybridIndex.load(path, schema, TestDocAdapter).use { reopened ->
            assertEquals(4L, reopened.count())
            val text = reopened.searchText(
                HybridTextQuery("swift actors", defaultFields = listOf("title", "body")),
                limit = 3,
            )
            assertEquals("swift-1", text.first().document.id)
            val hybrid = reopened.searchHybrid(
                query = HybridTextQuery("swift concurrency actors", defaultFields = listOf("title", "body")),
                embedding = TestEmbeddings.querySwift,
                limit = 3,
                efSearch = 50,
            )
            assertEquals("swift-1", hybrid.first().document.id)
        }
    }

    @Test
    fun dimensionMismatchIsTyped() = runTest {
        HybridIndex.create(tempPath(), schema, TestDocAdapter, config).use { index ->
            try {
                index.add(docs[0].first, FloatArray(64))
                fail("expected DimensionMismatch")
            } catch (e: HybridSearchException.DimensionMismatch) {
                assertEquals(128, e.expected)
                assertEquals(64, e.got)
            }
        }
    }

    @Test
    fun createTwiceThrowsIndexAlreadyExists() = runTest {
        val path = tempPath()
        // No commit: hnsw_rs cannot dump an empty graph (same limitation as the
        // Swift package); create() alone already persists hybrid.meta.json.
        HybridIndex.create(path, schema, TestDocAdapter, config).close()
        try {
            HybridIndex.create(path, schema, TestDocAdapter, config)
            fail("expected IndexAlreadyExists")
        } catch (_: HybridSearchException.IndexAlreadyExists) {
        }
    }
}
