package ai.botisan.hybridsearch

import ai.botisan.tantivy.TantivyDocumentAdapter
import ai.botisan.tantivy.TantivyDocumentWriter
import ai.botisan.tantivy.TantivyFieldMap
import ai.botisan.tantivy.tantivySchema
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/** On-device smoke: both native cores load and a hybrid round-trip works. */
@RunWith(AndroidJUnit4::class)
class HybridSmokeTest {

    private data class Note(val id: String, val text: String)

    private object NoteAdapter : TantivyDocumentAdapter<Note> {
        override fun encode(value: Note, doc: TantivyDocumentWriter) {
            doc.text("id", value.id)
            doc.text("text", value.text)
        }

        override fun decode(fields: TantivyFieldMap): Note =
            Note(fields.text("id")!!, fields.text("text")!!)
    }

    @Test
    fun hybridRoundTripOnDevice() = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.filesDir, "hybrid_smoke_${System.currentTimeMillis()}")
        val schema = tantivySchema {
            idField("id")
            textField("text")
        }
        HybridIndex.create(dir.absolutePath, schema, NoteAdapter, HybridIndexConfig(embeddingDimension = 4))
            .use { index ->
                index.index(Note("n1", "coffee at blue bottle"), floatArrayOf(1f, 0f, 0f, 0f))
                val hits = index.searchHybrid(
                    HybridTextQuery("coffee"),
                    floatArrayOf(1f, 0f, 0f, 0f),
                    limit = 1,
                )
                assertEquals("n1", hits.first().document.id)
            }
        dir.deleteRecursively()
        Unit
    }

    /** The full persistence cycle must run on the device's API level (metadata + vector files, API 24-safe writes). */
    @Test
    fun createCommitCloseReopenOnDevice() = runBlocking {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.filesDir, "hybrid_reopen_${System.currentTimeMillis()}")
        val schema = tantivySchema {
            idField("id")
            textField("text")
        }
        val config = HybridIndexConfig(embeddingDimension = 4)
        HybridIndex.create(dir.absolutePath, schema, NoteAdapter, config).use { index ->
            index.index(Note("n1", "coffee at blue bottle"), floatArrayOf(1f, 0f, 0f, 0f))
        }
        HybridIndex.open(dir.absolutePath, schema, NoteAdapter, config).use { reopened ->
            assertEquals(1L, reopened.count())
            val hits = reopened.searchHybrid(HybridTextQuery("coffee"), floatArrayOf(1f, 0f, 0f, 0f), limit = 1)
            assertEquals("n1", hits.first().document.id)
        }
        dir.deleteRecursively()
        Unit
    }
}
