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

    /** Every HNSW dump/sidecar file in the directory, any generation. */
    private fun hnswFiles(path: String): List<String> =
        File(path).listFiles().orEmpty().filter { it.isFile && it.name.startsWith("hnsw") }.map { it.name }.sorted()

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
        assertTrue(File(path, "hnsw-g1.hnsw.graph").exists())

        index.clear()
        assertEquals(emptyList<String>(), hnswFiles(path))
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

    /** Adapter whose encode signals `entered` and parks until `release` for the given doc id. */
    private class GatedAdapter(
        private val gatedId: String,
        val entered: CountDownLatch = CountDownLatch(1),
        val release: CountDownLatch = CountDownLatch(1),
    ) : TantivyDocumentAdapter<TestDoc> {
        override fun encode(value: TestDoc, doc: TantivyDocumentWriter) {
            if (value.id == gatedId) {
                entered.countDown()
                check(release.await(10, TimeUnit.SECONDS)) { "release latch timed out" }
            }
            TestDocAdapter.encode(value, doc)
        }

        override fun decode(fields: TantivyFieldMap): TestDoc = TestDocAdapter.decode(fields)
    }

    @Test
    fun indexCommitCannotSweepInAConcurrentUncommittedAdd() = runBlocking {
        val path = tempPath()
        val adapter = GatedAdapter("slow-1")

        HybridIndex.create(path, schema, adapter, config).use { index ->
            val slowDoc = TestDoc("slow-1", "Slow Doc", "held open by the test", true)
            val indexJob = launch(Dispatchers.Default) {
                index.index(slowDoc, TestEmbeddings.docSwiftConcurrency)
            }
            // Barrier: index() holds the mutex and is parked inside encode.
            assertTrue(adapter.entered.await(10, TimeUnit.SECONDS))
            val addJob = launch(Dispatchers.Default) {
                index.add(docs[1].document, docs[1].embedding)
            }
            // The mutex is held for index()'s whole add+commit, so the add
            // cannot have completed no matter how the scheduler runs it.
            delay(50)
            assertFalse(addJob.isCompleted)
            adapter.release.countDown()
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

    @Test
    fun deleteByFieldQueuesBehindACompoundIndexOperation() = runBlocking {
        val path = tempPath()
        val adapter = GatedAdapter("slow-2")

        HybridIndex.create(path, schema, adapter, config).use { index ->
            index.index(docs[0].document, docs[0].embedding) // "swift-1", committed
            val slowDoc = TestDoc("slow-2", "Slow Doc 2", "held open by the test", true)
            val indexJob = launch(Dispatchers.Default) {
                index.index(slowDoc, TestEmbeddings.docRustFfi)
            }
            assertTrue(adapter.entered.await(10, TimeUnit.SECONDS))
            val deleteJob = launch(Dispatchers.Default) {
                index.delete("id", TantivyValue.Text("swift-1"))
            }
            // Lookup + delete are one locked unit: they cannot interleave into
            // the in-flight index() while the encode gate is closed.
            delay(50)
            assertFalse(deleteJob.isCompleted)
            adapter.release.countDown()
            indexJob.join()
            deleteJob.join()

            assertNull(index.get("id", TantivyValue.Text("swift-1")))
            assertNotNull(index.get("id", TantivyValue.Text("slow-2")))
        }

        HybridIndex.load(path, schema, TestDocAdapter).use { reopened ->
            assertEquals(1L, reopened.count())
            assertNull(reopened.get("id", TantivyValue.Text("swift-1")))
            assertNotNull(reopened.get("id", TantivyValue.Text("slow-2")))
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
    fun adapterWritingDocIdFieldIsRejectedBeforeAnyStateChanges() = runTest {
        val path = tempPath()
        val evil = TestDoc("evil-1", "Evil", "writes the reserved field", true)
        val evilAdapter = object : TantivyDocumentAdapter<TestDoc> {
            override fun encode(value: TestDoc, doc: TantivyDocumentWriter) {
                TestDocAdapter.encode(value, doc)
                if (value.id == "evil-1") doc.u64("__doc_id", 999)
            }

            override fun decode(fields: TantivyFieldMap): TestDoc = TestDocAdapter.decode(fields)
        }
        HybridIndex.create(path, schema, evilAdapter, config).use { index ->
            repeat(3) {
                try {
                    index.add(evil, docs[0].embedding)
                    fail("expected ReservedField")
                } catch (_: HybridSearchException.ReservedField) {
                }
            }
            try {
                index.addAll(
                    listOf(
                        HybridDocument(docs[0].document, docs[0].embedding),
                        HybridDocument(evil, docs[1].embedding),
                    ),
                )
                fail("expected ReservedField")
            } catch (_: HybridSearchException.ReservedField) {
            }
            assertEquals(0L, index.count())
            index.commit()
            // Rejection happened before any FFI mutation: repeated programmer
            // errors left no (tombstoned) points — an empty commit writes no
            // dump files — and never advanced the id counter.
            assertEquals(emptyList<String>(), hnswFiles(path))
            assertEquals(0L, index.index(docs[0].document, docs[0].embedding))
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

    // -- HY2/HY3: the committed marker is authoritative ---------------------------------

    @Test
    fun missingGraphFileOnAPopulatedIndexIsCorruptionNotEmpty() = runTest {
        val path = tempPath()
        makeSeededIndex(path).first.close()
        assertTrue(File(path, "hnsw-g1.hnsw.graph").delete())

        try {
            HybridIndex.load(path, schema, TestDocAdapter)
            fail("expected VectorStateCorrupt")
        } catch (_: HybridSearchException.VectorStateCorrupt) {
        }
    }

    @Test
    fun partialVectorFilesOnAPopulatedIndexAreCorruption() = runTest {
        val path = tempPath()
        makeSeededIndex(path).first.close()
        assertTrue(File(path, "hnsw-g1.hnsw.data").delete()) // graph present, data gone

        try {
            HybridIndex.load(path, schema, TestDocAdapter)
            fail("expected VectorStateCorrupt")
        } catch (_: HybridSearchException.VectorStateCorrupt) {
        }
    }

    @Test
    fun strayVectorFilesUnderAnEmptyMarkerAreRemovedOnLoad() = runTest {
        val path = tempPath()
        HybridIndex.create(path, schema, TestDocAdapter, config).use { it.commit() }
        // Leftovers of an interrupted clear()/commit() — legacy-named or
        // generation-named: the committed marker says empty, so these must be
        // swept, not loaded.
        File(path, "hnsw.hnsw.graph").writeBytes(byteArrayOf(1, 2, 3))
        File(path, "hnsw.deleted").writeBytes(ByteArray(8))
        File(path, "hnsw-g7.hnsw.graph").writeBytes(byteArrayOf(4, 5, 6))

        HybridIndex.load(path, schema, TestDocAdapter).use { reopened ->
            assertEquals(0L, reopened.count())
            assertEquals(emptyList<String>(), hnswFiles(path))
            reopened.index(docs[0].document, docs[0].embedding)
            assertEquals(1L, reopened.count())
        }
    }

    @Test
    fun clearFailsLoudlyWhenVectorFilesCannotBeDeleted() = runTest {
        val path = tempPath()
        val (index, _) = makeSeededIndex(path)
        val dir = File(path)
        // A read-only directory denies unlink, so File.delete() reports false.
        check(dir.setWritable(false))
        try {
            try {
                index.clear()
                fail("expected VectorStateCorrupt")
            } catch (_: HybridSearchException.VectorStateCorrupt) {
            }
        } finally {
            check(dir.setWritable(true))
        }

        // Observable and recoverable: once deletion works again, clear() does too.
        index.clear()
        assertEquals(0L, index.count())
        assertEquals(emptyList<String>(), hnswFiles(path))
        index.close()
        HybridIndex.load(path, schema, TestDocAdapter).use { assertEquals(0L, it.count()) }
    }

    // -- HY2/HY3: crash-safe commit protocol -----------------------------------------

    @Test
    fun interruptedFirstCommitFailsLoadAsTornCommitInsteadOfSweepingTheGraph() = runTest {
        val path = tempPath()
        val meta = File(path, "hybrid.meta.json")
        val index = HybridIndex.create(path, schema, TestDocAdapter, config)
        val pristineMetadata = meta.readBytes() // what a crash before the publish leaves behind
        index.index(docs[0].document, docs[0].embedding)
        index.close()

        // Simulate the review scenario: Tantivy committed and the graph
        // dumped, but the crash struck before the metadata publish — the file
        // still says hasVectorGraph=false / nextDocId=0 / docCount=0.
        meta.writeBytes(pristineMetadata)
        try {
            HybridIndex.load(path, schema, TestDocAdapter)
            fail("expected TornCommit")
        } catch (_: HybridSearchException.TornCommit) {
        }
        // The graph was NOT deleted as a "stray" and no doc id was reused:
        // the failed load changed nothing on disk.
        assertTrue(File(path, "hnsw-g1.hnsw.graph").exists())
    }

    @Test
    fun countNeutralTornCommitIsDetectedByTheDocIdProbe() = runTest {
        val path = tempPath()
        val meta = File(path, "hybrid.meta.json")
        val index = HybridIndex.create(path, schema, TestDocAdapter, config)
        index.index(docs[0].document, docs[0].embedding) // docId 0
        val publishedMetadata = meta.readBytes() // nextDocId=1, docCount=1
        index.delete(0L) // committed count back to 0
        index.index(docs[1].document, docs[1].embedding) // docId 1, committed count 1 again
        index.close()

        // Committed count matches the stale metadata (1 == 1), so only the
        // __doc_id >= nextDocId probe can prove a later add was committed.
        meta.writeBytes(publishedMetadata)
        try {
            HybridIndex.load(path, schema, TestDocAdapter)
            fail("expected TornCommit")
        } catch (e: HybridSearchException.TornCommit) {
            assertTrue(e.message.orEmpty().contains("__doc_id"))
        }
    }

    @Test
    fun missingTombstoneSidecarOnAPopulatedIndexIsCorruption() = runTest {
        val path = tempPath()
        val (index, docIds) = makeSeededIndex(path)
        index.delete(docIds[0])
        index.close()
        // Losing the sidecar would resurrect the deleted vector (HnswIndex
        // treats a missing sidecar as "no tombstones"), so a marker-managed
        // index must refuse to load without it.
        assertTrue(File(path, "hnsw-g2.deleted").delete())

        try {
            HybridIndex.load(path, schema, TestDocAdapter)
            fail("expected VectorStateCorrupt")
        } catch (e: HybridSearchException.VectorStateCorrupt) {
            assertTrue(e.message.orEmpty().contains(".deleted"))
        }
    }

    @Test
    fun newGenerationStraysFromACrashedCommitAreSweptAndThePublishedGenerationLoads() = runTest {
        val path = tempPath()
        makeSeededIndex(path).first.close()
        // A crash after the generation-2 dump but before Tantivy's commit and
        // the metadata publish: generation 1 stays authoritative.
        File(path, "hnsw-g2.hnsw.graph").writeBytes(byteArrayOf(1, 2, 3))
        File(path, "hnsw-g2.hnsw.data").writeBytes(byteArrayOf(4, 5, 6))
        File(path, "hnsw-g2.deleted").writeBytes(ByteArray(8))

        HybridIndex.load(path, schema, TestDocAdapter).use { reopened ->
            assertEquals(4L, reopened.count())
            assertEquals(listOf("hnsw-g1.deleted", "hnsw-g1.hnsw.data", "hnsw-g1.hnsw.graph"), hnswFiles(path))
            assertEquals(
                "swift-1",
                reopened.searchVector(TestEmbeddings.querySwift, limit = 1, efSearch = 50).first().document.id,
            )
        }
    }

    // -- HY6: the adapter encodes each document exactly once ---------------------------

    @Test
    fun adapterEncodesEachDocumentExactlyOnceAcrossAddAndAddAll() = runTest {
        // Poison pill for a second encode: nothing in the adapter contract
        // promises purity, so an adapter may legally misbehave when re-invoked
        // for the same document — the old preflight+re-encode design turned
        // that into a tombstoned point and a burned id after passing
        // validation.
        val encodeCounts = mutableMapOf<String, Int>()
        val poisonOnSecondEncode = object : TantivyDocumentAdapter<TestDoc> {
            override fun encode(value: TestDoc, doc: TantivyDocumentWriter) {
                val calls = encodeCounts.merge(value.id, 1, Int::plus)!!
                TestDocAdapter.encode(value, doc)
                if (calls > 1) doc.u64("__doc_id", 999) // reserved — only reachable on a re-encode
            }

            override fun decode(fields: TantivyFieldMap): TestDoc = TestDocAdapter.decode(fields)
        }
        val path = tempPath()
        HybridIndex.create(path, schema, poisonOnSecondEncode, config).use { index ->
            assertEquals(0L, index.index(docs[0].document, docs[0].embedding))
            assertEquals(listOf(1L, 2L), index.addAll(listOf(docs[1], docs[2])))
            index.commit()
            assertEquals(mapOf("swift-1" to 1, "rust-1" to 1, "vector-1" to 1), encodeCounts)
            assertEquals(3L, index.count())
        }
    }

    // -- HY5: docId counter domain --------------------------------------------------

    private fun rewriteNextDocId(path: String, value: Long) {
        val meta = File(path, "hybrid.meta.json")
        val rewritten = meta.readText().replace("\"nextDocId\":0", "\"nextDocId\":$value")
        check(rewritten != meta.readText() || value == 0L) { "nextDocId not found in metadata" }
        meta.writeText(rewritten)
    }

    @Test
    fun negativeNextDocIdInMetadataFailsLoad() = runTest {
        val path = tempPath()
        HybridIndex.create(path, schema, TestDocAdapter, config).use { it.commit() }
        rewriteNextDocId(path, -5)

        try {
            HybridIndex.load(path, schema, TestDocAdapter)
            fail("expected MetadataCorrupt")
        } catch (e: HybridSearchException.MetadataCorrupt) {
            assertTrue(e.message.orEmpty().contains("nextDocId"))
        }
    }

    @Test
    fun docIdSpaceExhaustionFailsBeforeAnyMutation() = runTest {
        val path = tempPath()
        HybridIndex.create(path, schema, TestDocAdapter, config).use { it.commit() }
        rewriteNextDocId(path, Long.MAX_VALUE)

        HybridIndex.load(path, schema, TestDocAdapter).use { index ->
            suspend fun expectExhausted(block: suspend () -> Unit) {
                try {
                    block()
                    fail("expected IllegalStateException")
                } catch (e: IllegalStateException) {
                    assertTrue(e.message.orEmpty().contains("exhausted"))
                }
            }
            expectExhausted { index.add(docs[0].document, docs[0].embedding) }
            expectExhausted { index.addAll(docs) }
            assertEquals(0L, index.count())
            index.commit()
        }
        // The failed adds mutated nothing: no documents, no vector files.
        HybridIndex.load(path, schema, TestDocAdapter).use { reopened ->
            assertEquals(0L, reopened.count())
            assertEquals(emptyList<String>(), hnswFiles(path))
        }
    }

    @Test
    fun lastMintableDocIdIsUsableAndTheCounterNeverWraps() = runTest {
        val path = tempPath()
        HybridIndex.create(path, schema, TestDocAdapter, config).use { it.commit() }
        rewriteNextDocId(path, Long.MAX_VALUE - 1)

        HybridIndex.load(path, schema, TestDocAdapter).use { index ->
            // A batch that would need MAX itself is rejected before mutating...
            try {
                index.addAll(listOf(docs[0], docs[1]))
                fail("expected IllegalStateException")
            } catch (_: IllegalStateException) {
            }
            assertEquals(0L, index.count())
            // ...while the last representable single mint works and the
            // counter lands exactly on MAX without wrapping.
            assertEquals(Long.MAX_VALUE - 1, index.index(docs[0].document, docs[0].embedding))
            try {
                index.add(docs[1].document, docs[1].embedding)
                fail("expected IllegalStateException")
            } catch (_: IllegalStateException) {
            }
            assertEquals(1L, index.count())
        }
        HybridIndex.load(path, schema, TestDocAdapter).use { reopened ->
            assertEquals(1L, reopened.count())
            assertNotNull(reopened.get(Long.MAX_VALUE - 1))
        }
    }

    // -- HY4: delete-by-field addresses records without decoding them ----------------

    @Test
    fun deleteByFieldDoesNotRequireUserDocumentDecoding() = runTest {
        val path = tempPath()
        HybridIndex.create(path, schema, TestDocAdapter, config).use { index ->
            index.index(docs[0].document, docs[0].embedding)
        }

        val brokenDecoder = object : TantivyDocumentAdapter<TestDoc> {
            override fun encode(value: TestDoc, doc: TantivyDocumentWriter) = TestDocAdapter.encode(value, doc)

            override fun decode(fields: TantivyFieldMap): TestDoc =
                error("legacy record this decoder cannot read")
        }
        HybridIndex.load(path, schema, brokenDecoder).use { index ->
            // The decoder really is broken for full document reads...
            try {
                index.get("id", TantivyValue.Text("swift-1"))
                fail("expected the decoder failure")
            } catch (e: IllegalStateException) {
                assertTrue(e.message.orEmpty().contains("legacy record"))
            }
            // ...but delete-by-field resolves only the internal stored id.
            index.delete("id", TantivyValue.Text("swift-1"))
            assertEquals(0L, index.count())
        }

        HybridIndex.load(path, schema, TestDocAdapter).use { assertEquals(0L, it.count()) }
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
