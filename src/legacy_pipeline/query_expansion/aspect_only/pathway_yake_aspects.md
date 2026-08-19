# Aspect Extraction: YAKE Pathway (Statistical & Vector)

## Objective
To deterministically parse a complex user query into a weighted list of orthogonal core aspects using local heuristics (Regex + YAKE) at sub-10ms latency. **This pathway performs zero keyword expansion.** 

This core extraction pipeline powers both the Statistical and Vector approaches. They share the same extraction logic but differ in how they calculate the final aspect `weight`.

## Core Principles
1. **Acronym Preservation:** Statistical extractors often drop short, vital acronyms (e.g., "AI", "API"). We must forcibly recover them.
2. **Lexical Subsumption:** Long multi-word phrases should consume shorter substrings to prevent redundant, overlapping aspects.
3. **Purity:** Stop words must be filtered out immediately.

---

## Step-by-Step Flow

### Step 1: Pre-processing & Acronym Recovery
Before statistical extraction, we must ensure critical short terms are not lost.
1. Clean the query (remove punctuation, lowercase standard words).
2. **Regex Acronym Catch:** Extract any fully capitalized word of 2+ characters (e.g., `\b[A-Z]{2,}\b`). 
3. Store these recovered acronyms (e.g., "AI") in a `must_include` list.

### Step 2: YAKE N-Gram Extraction
Run the query through the YAKE (Yet Another Keyword Extractor) algorithm.
- **n:** 3 (Extract unigrams, bigrams, and trigrams).
- **Stopword Filter:** Apply a strict English stopword list *before* YAKE processes the text so it does not build phrases around "with" or "that".
- **Output:** A list of raw n-grams sorted by statistical importance.

### Step 3: Aspect Merging
Combine the `must_include` list (from Step 1) with the top N results from YAKE.

### Step 4: Global Lexical Subsumption (Orthogonalization)
To prevent overlapping aspects (e.g., "Digital Experience" vs "Experience"), apply lexical subsumption.
1. Sort the merged aspects by string length (longest first).
2. For each aspect, if it is completely contained as a substring within a longer, already-accepted aspect, **discard it**.
3. **Output:** A clean list of mutually exclusive string aspects.

---

## Step 5: Weight Calculation (The Two Variants)

At this point, we have a list of orthogonal aspects (e.g., `["digital experience", "ai"]`). We must now calculate a `weight` (0.0 to 1.0) for each.

### Variant A: Statistical Weighting
Used by the Statistical Pipeline. Importance is determined by **global rarity**.
1. Look up each word in the aspect against the **Global IDF Dictionary**.
2. Compute the average IDF of the words in the aspect.
3. Normalize the IDF score using a softmax or Min-Max scaler so the highest-IDF aspect gets `1.0`.
*Logic: Rarer terms (like "genstudio") are more critical constraints than common terms (like "integrate").*

### Variant B: Vector Weighting
Used by the Vector Pipeline. Importance is determined by **semantic centrality**.
1. Encode the original full query into a vector $V_Q$ using a local embedding model (e.g., BGE-Small).
2. Encode each extracted aspect into a vector $V_{A_i}$.
3. Calculate the Cosine Similarity between $V_Q$ and each $V_{A_i}$.
4. Normalize the similarities so the highest similarity gets `1.0`.
*Logic: Aspects that most closely align with the overall semantic meaning of the full query receive the highest weight.*

## Example Input / Output

**Input Query:** "How does Adobe integrate AI into its Digital Experience segment?"

**Expected Final Output:**
```python
[
    ("digital experience segment", 1.0), # (Weight will vary slightly depending on Variant A vs B)
    ("ai", 0.95),                        # (Recovered via Regex in Step 1)
    ("adobe integrate", 0.4)
]
```
