package ai.botisan.hybridsearch

import ai.botisan.tantivy.TantivyDocumentAdapter
import ai.botisan.tantivy.TantivyDocumentWriter
import ai.botisan.tantivy.TantivyFieldMap
import ai.botisan.tantivy.TantivyQuery
import ai.botisan.tantivy.TantivyValue
import ai.botisan.tantivy.tantivySchema
import java.io.File
import java.nio.file.Files
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * Port of HybridSearchSwiftTests (HybridSearch.swift @ 0.1.1) plus the #195
 * review's regression matrix (durable empty/clear state, compound-op locking,
 * validation, reserved fields, provably exclusive fusion legs).
 */
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
        HybridDocument(TestDoc("swift-1", "Swift Concurrency", "Async await and actors in Swift.", true), TestEmbeddings.docSwiftConcurrency),
        HybridDocument(TestDoc("rust-1", "Rust FFI", "Calling Rust from Swift using UniFFI.", true), TestEmbeddings.docRustFfi),
        HybridDocument(TestDoc("vector-1", "Vector Search with HNSW", "Approximate nearest neighbor search using HNSW graphs.", false), TestEmbeddings.docVectorHnsw),
        HybridDocument(TestDoc("tantivy-1", "Full-text Search with Tantivy", "Indexing and BM25 ranking with Tantivy.", true), TestEmbeddings.docTantivy),
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
    fun dimensionMismatchIsTypedAndRolledBack() = runTest {
        HybridIndex.create(tempPath(), schema, TestDocAdapter, config).use { index ->
            try {
                index.add(docs[0].document, FloatArray(64))
                fail("expected DimensionMismatch")
            } catch (e: HybridSearchException.DimensionMismatch) {
                assertEquals(128, e.expected)
                assertEquals(64, e.got)
            }
            assertEquals(0L, index.count())
            index.index(docs[0].document, docs[0].embedding)
            assertEquals(1L, index.count())
        }
    }

    @Test
    fun createTwiceThrowsIndexAlreadyExists() = runTest {
        val path = tempPath()
        HybridIndex.create(path, schema, TestDocAdapter, config).use { index ->
            index.commit() // empty commit is supported: no vector files, tantivy + metadata committed
        }
        try {
            HybridIndex.create(path, schema, TestDocAdapter, config)
            fail("expected IndexAlreadyExists")
        } catch (_: HybridSearchException.IndexAlreadyExists) {
        }
    }

    // -- HY3: empty indexes must reopen -----------------------------------------

    @Test
    fun createCloseReopenEmptyIndexWorks() = runTest {
        val path = tempPath()
        // No commit at all — simulates closing (or crashing) before the first commit.
        HybridIndex.create(path, schema, TestDocAdapter, config).close()

        HybridIndex.open(path, schema, TestDocAdapter, config).use { reopened ->
            assertEquals(0L, reopened.count())
            assertTrue(reopened.searchText(HybridTextQuery("anything"), limit = 3).isEmpty())
            assertTrue(reopened.searchVector(TestEmbeddings.queryVector, limit = 3, efSearch = 50).isEmpty())

            reopened.index(docs[0].document, docs[0].embedding)
            assertEquals(1L, reopened.count())
            assertEquals(
                "swift-1",
                reopened.searchVector(TestEmbeddings.querySwift, limit = 1, efSearch = 50).first().document.id,
            )
        }
    }

    // -- HY2: clear() must be durable --------------------------------------------

    @Test
    fun clearRemovesVectorFilesAndSurvivesReopen() = runTest {
        val path = tempPath()
        val (index, _) = makeSeededIndex(path)
        assertTrue(File(path, "hnsw.hnsw.graph").exists())

        index.clear()
        assertFalse(File(path, "hnsw.hnsw.graph").exists())
        assertFalse(File(path, "hnsw.hnsw.data").exists())
        assertFalse(File(path, "hnsw.deleted").exists())
        assertEquals(0L, index.count())
        index.close()

        HybridIndex.load(path, schema, TestDocAdapter).use { reopened ->
            assertEquals(0L, reopened.count())
            assertTrue(reopened.searchText(HybridTextQuery("swift"), limit = 5).isEmpty())
            assertTrue(reopened.searchVector(TestEmbeddings.queryVector, limit = 3, efSearch = 50).isEmpty())

            // Doc ids restart at 0 and both legs work again after re-adding.
            val newIds = reopened.addAll(docs)
            reopened.commit()
            assertEquals(listOf(0L, 1L, 2L, 3L), newIds)
            assertEquals(
                "swift-1",
                reopened.searchText(HybridTextQuery("swift actors", defaultFields = listOf("title", "body")), limit = 3)
                    .first().document.id,
            )
            assertTrue(reopened.searchVector(TestEmbeddings.querySwift, limit = 3, efSearch = 50).isNotEmpty())
        }
    }

    // -- HY4: compound operations hold one lock -----------------------------------

    @Test
    fun indexCommitCannotSweepInAConcurrentUncommittedAdd() = runBlocking {
        val path = tempPath()
        val slowGate = CountDownLatch(1)
        val slowAdapter = object : TantivyDocumentAdapter<TestDoc> {
            override fun encode(value: TestDoc, doc: TantivyDocumentWriter) {
                if (value.id == "slow-1") slowGate.await(5, TimeUnit.SECONDS)
                TestDocAdapter.encode(value, doc)
            }

            override fun decode(fields: TantivyFieldMap): TestDoc = TestDocAdapter.decode(fields)
        }

        HybridIndex.create(path, schema, slowAdapter, config).use { index ->
            val slowDoc = TestDoc("slow-1", "Slow Doc", "held open by the test", true)
            val indexJob = launch(Dispatchers.Default) {
                index.index(slowDoc, TestEmbeddings.docSwiftConcurrency)
            }
            delay(100) // let index() take the lock and park inside encode
            val addJob = launch(Dispatchers.Default) {
                index.add(docs[1].document, docs[1].embedding) // must queue behind index()'s whole add+commit
            }
            delay(100)
            slowGate.countDown()
            indexJob.join()
            addJob.join()
        }

        // Only the index()'ed doc may be committed; the concurrent add() stayed
        // uncommitted and is gone after reopen.
        HybridIndex.load(path, schema, TestDocAdapter).use { reopened ->
            assertEquals(1L, reopened.count())
            assertNotNull(reopened.get("id", TantivyValue.Text("slow-1")))
            assertNull(reopened.get("id", TantivyValue.Text("rust-1")))
        }
    }

    // -- HY5: validation -----------------------------------------------------------

    @Test
    fun configValidationRejectsOutOfRangeValues() {
        fun expectInvalid(block: () -> Unit) {
            try {
                block()
                fail("expected IllegalArgumentException")
            } catch (_: IllegalArgumentException) {
            }
        }
        expectInvalid { HybridIndexConfig(embeddingDimension = 0) }
        expectInvalid { HybridIndexConfig(embeddingDimension = -4) }
        expectInvalid { HybridIndexConfig(hnswMaxConnections = 0) }
        expectInvalid { HybridIndexConfig(hnswMaxConnections = 257) }
        expectInvalid { HybridIndexConfig(hnswMaxElements = 0) }
        expectInvalid { HybridIndexConfig(hnswMaxLayers = 17) }
        expectInvalid { HybridIndexConfig(hnswEfConstruction = 0) }
    }

    @Test
    fun searchArgumentValidation() = runTest {
        val (index, _) = makeSeededIndex()
        index.use {
            suspend fun expectInvalid(block: suspend () -> Unit) {
                try {
                    block()
                    fail("expected IllegalArgumentException")
                } catch (_: IllegalArgumentException) {
                }
            }
            expectInvalid { it.searchText(HybridTextQuery("x"), limit = 0) }
            expectInvalid { it.searchText(HybridTextQuery("x"), limit = -1) }
            expectInvalid { it.searchText(HybridTextQuery("x"), limit = 5, offset = -1) }
            expectInvalid { it.searchVector(TestEmbeddings.queryVector, efSearch = 0) }
            expectInvalid { it.searchVector(TestEmbeddings.queryVector, overfetchMultiplier = 0) }
            expectInvalid { it.searchHybrid(HybridTextQuery("x"), TestEmbeddings.queryVector, rrfK = Float.NaN) }
            expectInvalid { it.searchHybrid(HybridTextQuery("x"), TestEmbeddings.queryVector, rrfK = 0f) }
            expectInvalid { it.searchHybrid(HybridTextQuery("x"), TestEmbeddings.queryVector, textWeight = -1f) }
            expectInvalid {
                it.searchHybrid(HybridTextQuery("x"), TestEmbeddings.queryVector, textWeight = 0f, vectorWeight = 0f)
            }
        }
    }

    // -- HY6: reserved field + primary id invariants --------------------------------

    @Test
    fun schemaDeclaringDocIdFieldIsRejected() = runTest {
        val badSchema = tantivySchema {
            idField("id")
            u64Field("__doc_id")
        }
        try {
            HybridIndex.create(tempPath(), badSchema, TestDocAdapter, config)
            fail("expected ReservedField")
        } catch (e: HybridSearchException.ReservedField) {
            assertEquals("__doc_id", e.name)
        }
    }

    @Test
    fun adapterWritingDocIdFieldIsRejectedAndRolledBack() = runTest {
        val evilAdapter = object : TantivyDocumentAdapter<TestDoc> {
            override fun encode(value: TestDoc, doc: TantivyDocumentWriter) {
                TestDocAdapter.encode(value, doc)
                doc.u64("__doc_id", 999)
            }

            override fun decode(fields: TantivyFieldMap): TestDoc = TestDocAdapter.decode(fields)
        }
        HybridIndex.create(tempPath(), schema, evilAdapter, config).use { index ->
            try {
                index.add(docs[0].document, docs[0].embedding)
                fail("expected ReservedField")
            } catch (_: HybridSearchException.ReservedField) {
            }
            assertEquals(0L, index.count())
        }
    }

    @Test
    fun documentsMustCarryExactlyOnePrimaryIdValue() = runTest {
        val missingIdAdapter = object : TantivyDocumentAdapter<TestDoc> {
            override fun encode(value: TestDoc, doc: TantivyDocumentWriter) {
                doc.text("title", value.title) // never writes "id"
            }

            override fun decode(fields: TantivyFieldMap): TestDoc = TestDocAdapter.decode(fields)
        }
        HybridIndex.create(tempPath(), schema, missingIdAdapter, config).use { index ->
            try {
                index.add(docs[0].document, docs[0].embedding)
                fail("expected InvalidPrimaryIdValue")
            } catch (e: HybridSearchException.InvalidPrimaryIdValue) {
                assertEquals(0, e.count)
            }
            assertEquals(0L, index.count())
        }

        val doubleIdAdapter = object : TantivyDocumentAdapter<TestDoc> {
            override fun encode(value: TestDoc, doc: TantivyDocumentWriter) {
                TestDocAdapter.encode(value, doc)
                doc.text("id", "${value.id}-again")
            }

            override fun decode(fields: TantivyFieldMap): TestDoc = TestDocAdapter.decode(fields)
        }
        HybridIndex.create(tempPath(), schema, doubleIdAdapter, config).use { index ->
            try {
                index.add(docs[0].document, docs[0].embedding)
                fail("expected InvalidPrimaryIdValue")
            } catch (e: HybridSearchException.InvalidPrimaryIdValue) {
                assertEquals(2, e.count)
            }
        }
    }

    // -- HY8: fusion with provably exclusive legs ------------------------------------

    @Test
    fun hybridFusesLegsThatAreProvablyExclusive() = runTest {
        val dim4 = HybridIndexConfig(embeddingDimension = 4, hnswMaxElements = 1_000)
        HybridIndex.create(tempPath(), schema, TestDocAdapter, dim4).use { index ->
            fun v(x: Float, y: Float) = floatArrayOf(x, y, 0f, 0f)

            val lexOnly = TestDoc("lex-1", "zebra crossing", "the zebra token appears only here", true)
            val vecOnly = TestDoc("vec-1", "unrelated title", "no matching words at all", true)
            val fillers = (1..8).map { TestDoc("filler-$it", "filler title", "filler body words", true) }

            index.addAll(
                listOf(
                    HybridDocument(lexOnly, v(0f, 1f)), // orthogonal to the query vector -> farthest
                    HybridDocument(vecOnly, v(1f, 0.05f)), // nearest to the query vector
                ) + fillers.mapIndexed { i, d -> HybridDocument(d, v(1f, 0.1f + i * 0.02f)) },
            )
            index.commit()

            val queryVec = v(1f, 0f)
            // searchHybrid(limit = 2) fetches 6 candidates per leg — query each leg
            // independently at that same depth and prove exclusivity first.
            val textIds = index.searchText("zebra", limit = 6).map { it.document.id }
            assertEquals(listOf("lex-1"), textIds)

            val vectorIds = index.searchVector(queryVec, limit = 6, efSearch = 50).map { it.document.id }
            assertTrue("vector leg unexpectedly contains the lexical-only doc: $vectorIds", "lex-1" !in vectorIds)
            assertTrue("vector leg is missing the vector-only doc: $vectorIds", "vec-1" in vectorIds)

            val fused = index.searchHybrid("zebra", queryVec, limit = 2, efSearch = 50).map { it.document.id }
            assertTrue("lexical-only hit missing from fusion: $fused", "lex-1" in fused)
            assertTrue("vector-only hit missing from fusion: $fused", "vec-1" in fused)
        }
    }

    // -- lifecycle ---------------------------------------------------------------------

    @Test
    fun postCloseCallsThrowAlreadyClosed() = runTest {
        val (index, _) = makeSeededIndex()
        index.close()
        index.close() // idempotent
        try {
            index.count()
            fail("expected AlreadyClosed")
        } catch (_: HybridSearchException.AlreadyClosed) {
        }
        try {
            index.searchText(HybridTextQuery("x"), limit = 1)
            fail("expected AlreadyClosed")
        } catch (_: HybridSearchException.AlreadyClosed) {
        }
    }
}
