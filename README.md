# HybridSearch.kt

Hybrid search for **Android/Kotlin**: BM25 full-text ([tantivy.kt](https://github.com/botisan-ai/tantivy.kt)) + HNSW vector search ([HNSW.kt](https://github.com/lhr0909/HNSW.kt)) fused with weighted **reciprocal-rank fusion**.

Pure-Kotlin port of [HybridSearch.swift](https://github.com/botisan-ai/HybridSearch.swift)'s `HybridIndex` actor — same architecture (documents live in Tantivy, vectors in HNSW keyed by a minted `__doc_id`, `hybrid.meta.json` metadata) with two documented deviations:

1. **Deterministic RRF tie-break** (score desc, then docId asc). The Swift implementation's dictionary-order tie-break is nondeterministic.
2. **Schema fingerprint** derives from the explicit Kotlin schema DSL, not Swift runtime reflection — index directories are not cross-language portable (they were never shared across devices anyway).

## Install

GitHub Release assets per version (see the tantivy.kt README for the repo-setup snippet):

- `hybridsearch-android-<version>-maven.zip` — Maven-layout repo whose zip root **is** the repository root (`ai/botisan/...`); unzipping it (plus the tantivy/hnsw zips) and pointing Gradle at the folders resolves everything transitively.
- `hybridsearch-android-<version>.aar` + `.sha256` files.

```kotlin
dependencies { implementation("ai.botisan:hybridsearch-android:<version>") }
```

## Usage

```kotlin
import ai.botisan.hybridsearch.*
import ai.botisan.tantivy.*

data class Receipt(val id: String, val merchant: String, val notes: String?)

val schema = tantivySchema {
    idField("id")
    textField("merchant")          // unicode tokenizer by default (CJK-friendly)
    textField("notes")
    facetField("tagIds")
}

object ReceiptAdapter : TantivyDocumentAdapter<Receipt> {
    override fun encode(value: Receipt, doc: TantivyDocumentWriter) {
        doc.text("id", value.id)
        doc.text("merchant", value.merchant)
        value.notes?.let { doc.text("notes", it) }
    }
    override fun decode(fields: TantivyFieldMap) =
        Receipt(fields.text("id")!!, fields.text("merchant")!!, fields.text("notes"))
}

HybridIndex.open(dir, schema, ReceiptAdapter, HybridIndexConfig(embeddingDimension = 256))
    .use { index ->                                           // dir: File or String
        index.index(receipt, embedding)                       // add + commit, one lock
        index.addAll(receipts.map { HybridDocument(it, embeddingFor(it)) })
        index.searchText("coffee")                            // lexical only — no embedding needed
        index.searchVector(queryEmbedding)                    // vector only
        index.searchHybrid("coffee", queryEmbedding)          // RRF fusion
        index.delete("id", TantivyValue.Text(receipt.id))
    }
```

Embeddings are **caller-supplied** (`FloatArray` of the configured dimension) — this package never runs a model. Defaults mirror the Swift package: 384-d cosine, `rrfK = 60`, both weights 1, overfetch ×3, `efSearch` floored at the fetch limit.

Contract notes:

- **Every public operation holds one lock**, including compound ones (`index` = add + commit, delete-by-field = lookup + delete), so concurrent callers cannot interleave between their steps.
- **Durable empty state:** missing HNSW files mean an empty vector graph. `commit()` on an empty index works (skips the vector dump — `hnsw_rs` cannot dump zero points — and removes stale files), `clear()` deletes the vector files, and create → close → reopen round-trips fine.
- `__doc_id` is reserved: declaring it in a schema or writing it from an adapter throws `ReservedField`. Every encoded document must carry exactly one value for the primary id field (`InvalidPrimaryIdValue` otherwise).
- Config bounds are enforced at construction (dimension/capacities/connections/layers); paging, `efSearch`, overfetch, `rrfK` and weights are validated per call with `IllegalArgumentException`.
- `close()` waits for in-flight operations; afterwards calls throw `AlreadyClosed`.

## Development

```bash
./bootstrap-deps.sh                 # fetch sibling release repos (or build ../tantivy.kt + ../HNSW.kt locally)
cd android && ./gradlew test        # host-JVM suite (needs sibling checkouts' host dylibs for JNA)
cd android && ./gradlew lintRelease # release lint gate (also runs inside gh-release.sh)
./gh-release.sh                     # test + lint + assemble + GitHub Release with maven.zip + aar + sha256
```

Host-JVM tests load the sibling repos' host dylibs via `jna.library.path`; build them once with `cargo build --release` in each sibling's `rust/`.

## License

MIT — see [LICENSE](LICENSE).
