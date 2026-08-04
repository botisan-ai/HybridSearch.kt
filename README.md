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

- **Every public operation holds one lock**, including compound ones (`index` = add + commit, delete-by-field = lookup + delete), so concurrent callers cannot interleave between their steps. Delete-by-field resolves only the internal stored id — it works even when the user decoder cannot read a record.
- **Crash-safe commits, verified on load:** each commit dumps vectors under a fresh `hnsw-g<generation>` basename, commits Tantivy, atomically publishes metadata naming the generation (plus the committed `docCount`), then sweeps older generations — so published metadata always points at a complete dump and a crash mid-cycle leaves only sweepable strays. `load()` cross-checks the committed Tantivy state (docCount, and a probe for any `__doc_id >= nextDocId`) and fails with `TornCommit` when a crash struck between Tantivy's commit and the publish — vectors are not reconstructable, so ids are never silently reused. An empty generation has no HNSW files (`hnsw_rs` cannot dump zero points; `hasVectorGraph` records which state to expect); missing/partial files under a populated marker — the `.deleted` tombstone sidecar included — fail with `VectorStateCorrupt`, never a silent downgrade to text-only search. The sweep touches only files the package owns (the legacy dump names and `hnsw-g<n>` triplets — never arbitrary `hnsw*`-prefixed files kept in the directory), and it fails loudly rather than pretending: an undeletable stale file or an unenumerable directory makes `load()`/`clear()`/`commit()` throw instead of resurrecting state on reopen.
- `__doc_id` is reserved: declaring it in a schema or writing it from an adapter throws `ReservedField`. Every encoded document must carry exactly one value for the primary id field (`InvalidPrimaryIdValue` otherwise). The adapter encodes each document **exactly once** — the Tantivy write's own encode is the validation, and it runs before the document reaches any FFI or the vector graph, so a rejected document leaves no tombstoned vector and does not advance the id counter (stateful adapters cannot be re-invoked into a different answer). Batches are all-or-nothing down to native validation (a bad facet path or JSON string in any document rejects the whole batch with nothing staged), and **a failed add publishes nothing**: when the vector leg fails after the text write, the rollback masks that text inside the same open transaction — it never commits, so other callers' still-uncommitted work stays uncommitted.
- Config bounds are enforced at construction (dimension, capacities ≤ 16,777,216, connections 2..255, layers ≤ 16); paging, `efSearch`, overfetch, `rrfK` and weights are validated per call with `IllegalArgumentException`. `nextDocId` and the commit `generation` are validated on load (`MetadataCorrupt` when negative) and advanced with checked arithmetic. Every publishing entry point — `index`, `indexAll`, direct `commit`, persistent delete (both overloads), `compact`, and `clear` — reserves its generation under the outer lock before its first text, vector, or ID mutation and passes that value through the commit; exhaustion therefore leaves both in-memory and durable state unchanged.
- `close()` waits for in-flight operations; afterwards calls throw `AlreadyClosed`.

## Development

```bash
./bootstrap-deps.sh                 # fetch sibling release repos (or build ../tantivy.kt + ../HNSW.kt locally)
cd android && ./gradlew test        # host-JVM suite (needs sibling checkouts' host dylibs for JNA)
cd android && ./gradlew lintRelease # release lint gate (also runs inside gh-release.sh)
./gh-release.sh                     # bootstrap released deps -> test + lint + assemble -> consumer-resolution
                                    # gate (the staged zip + downloaded dep repos resolved by a cache-isolated
                                    # generated AGP consumer) -> GitHub Release with maven.zip + aar + sha256
```

Host-JVM tests load the sibling repos' host dylibs via `jna.library.path`; build them once with `cargo build --release` in each sibling's `rust/`.

## License

MIT — see [LICENSE](LICENSE).
