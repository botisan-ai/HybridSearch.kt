package ai.botisan.hybridsearch

import ai.botisan.tantivy.TantivyQuery

/**
 * Text leg of a hybrid search. A blank [query] means "match all" (port of
 * HybridSearch.swift's `HybridTextQuery`).
 */
public data class HybridTextQuery(
    val query: String,
    val defaultFields: List<String> = emptyList(),
    val fuzzyFields: List<TantivyQuery.FuzzyField> = emptyList(),
) {
    internal fun toTantivyQuery(defaultFieldsFallback: List<String>): TantivyQuery {
        if (query.isBlank()) return TantivyQuery.All
        val fields = defaultFields.ifEmpty { defaultFieldsFallback }
        return TantivyQuery.QueryString(query, fields, fuzzyFields)
    }
}
