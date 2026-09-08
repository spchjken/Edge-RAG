# Implementation Plan — Corpus-Informed Query Expansion Research Plan Revision

Status: executed and verified on 2026-09-09.

## Objective

Refine `docs/corpus_informed_query_expansion_plan.md` around the primary proposed method discussed
with the user: capacity-bounded, corpus-specific term-context memory for training-free query
expansion before a single BM25 retrieval.

## Approved scope

- Reposition V7 as an archived implementation/control rather than an active experiment.
- Make context-reranked vocabulary expansion the central proposed method.
- Cap the semantic vocabulary at approximately 10,000–15,000 terms.
- Compare salience, semantic coverage and hybrid vocabulary construction.
- Specify deterministic collection of up to 20 short corpus contexts per selected term.
- Separate corpus-representation quality, query-context compatibility and actual expansion utility.
- Compare max, top-2 mean and support-aware context aggregation.
- Keep all query-time neural work vector-only by pre-encoding contexts at index time.
- Preserve explicit expansion-mass and postings-cost limits.
- Add storage, construction and query-latency accounting.
- Replace obsolete analyzed/unified-RM3 future controls with the current Terrier baseline suite.
- State conservative novelty boundaries relative to static embedding QE, CEQE, PRF and generated QE.
- Move graphs, topics, relation inventories and similar structures to later extensions.

## Verification

- Review the revised document for internal consistency and stale V7 claims.
- Check all repository-relative evidence links used by the revised plan.
- Inspect the final diff without modifying unrelated user changes.
