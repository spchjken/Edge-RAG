# Routing Architecture: Cost-Aware Cascade (Mass × Focus)

## Objective
To efficiently triage lexical search hits into three discrete buckets (**Bypass**, **Rerank**, **Discard**) by decoupling the density score into two mathematically orthogonal signals: **Mass** (total weighted footprint) and **Focus** (concentration of the longest unbroken span).

This replaces the legacy additive model ($\rho_{cont} + \rho_{scat}$) which, after simplification, collapsed to a single magnitude axis: $\rho_{scat}(1 + \phi)$, where $\phi = \rho_{cont}/\rho_{scat}$. That formula cannot express "either a long total footprint OR a long unbroken span is sufficient," because $\rho_{cont} \le \rho_{scat}$ always holds by construction. Separating Mass from Focus recovers the second degree of freedom.

By framing this as a **Cost-Aware Cascade** (Wang et al., SIGIR 2011; Cambazoglu et al., WSDM 2010), we position the CPU router as the cheap early stage that promotes obvious positives (Bypass) and prunes obvious negatives (Discard), spending the expensive LLM reranker only on the genuinely ambiguous middle band.

---

## Definitions

Let chunk $c_i$ have character length $|c_i|$. After Aho-Corasick matching and interval merging, the chunk contains a set of disjoint merged intervals $\{m_1, m_2, \ldots, m_n\}$.

### Weight Aggregation ($\bar{w}_x$)
A merged interval $m_x$ may absorb hits from multiple keywords across multiple aspects. We define the effective weight of interval $m_x$ as:

$$\bar{w}_x = \max_{k \in m_x} w_k$$

where $w_k$ is the weight of keyword $k$ that contributed a hit inside $m_x$. Using $\max$ ensures the interval is valued by its strongest contributing signal (not inflated by accumulating weak hits).

### Aspect Coverage ($\alpha$)
The fraction of the query's total aspect weight that has at least one match in the chunk:

$$\alpha(c_i) = \frac{\sum_{j \in \mathcal{A}_{found}(c_i)} w_j}{\sum_{j \in \mathcal{A}_{query}} w_j} \in [0, 1]$$

where $\mathcal{A}_{query}$ is the full set of query aspects (each with weight $w_j$) and $\mathcal{A}_{found}(c_i)$ is the subset of aspects that produced at least one hit in chunk $c_i$.

---

## Step-by-Step Flow

### Step 1: Orthogonal Signal Extraction
For each chunk $c_i$, compute two independent features from its merged intervals:

**1. Mass ($\mu$) — "Total Density"**
The proportion of the chunk covered by weighted aspect matches. Since intervals are disjoint ($\sum_x |m_x| \le |c_i|$) and $\bar{w}_x \in [0, 1]$, we have $\mu \in [0, 1]$ naturally.

$$\mu(c_i) = \frac{1}{|c_i|} \sum_x |m_x| \cdot \bar{w}_x$$

*Captures: "Is the chunk densely packed with scattered relevance across its full length?"*

**2. Focus ($\phi$) — "Unbroken Span Concentration"**
The share of total weighted mass held by the single longest (heaviest) interval. This is a pure shape statistic, decorrelated from the magnitude $\mu$.

$$\phi(c_i) = \frac{\max_x \bigl(|m_x| \cdot \bar{w}_x\bigr)}{\sum_x \bigl(|m_x| \cdot \bar{w}_x\bigr)} \in (0, 1]$$

*Captures: "Is one contiguous passage dominating all the evidence?"*

**Edge case:** If there are no matched intervals ($n = 0$), both $\mu$ and $\phi$ are defined as $0$, and the chunk routes directly to Discard.

### Step 2: Three-Way Triage (Cascade Decision)
Instead of a single threshold producing a two-way split (bypass vs. everything else), we apply **two-sided thresholding** to produce a three-way confidence triage:

| Region | Meaning | Action | LLM KV Cost |
|---|---|---|---|
| Confidently relevant | Density so high that LLM confirmation is redundant | **Bypass** → `Bypass_List` | 0 |
| Genuinely uncertain | Medium / fragmented evidence | **Rerank** → `Rerank_Queue` | Paid |
| Confidently irrelevant | Almost no weighted matches | **Discard** | 0 |

**OR-Gate Formulation:**
Bypass requires adequate aspect coverage **AND** either sufficient total density **OR** a dominant focused passage:

$$
\text{route}(c_i) =
\begin{cases}
\textbf{Bypass} & \text{if } \alpha(c_i) \ge \tau_{\alpha} \text{ and } \bigl(\mu(c_i) \ge \tau_{\mu} \text{ or } \phi(c_i) \ge \tau_{\phi}\bigr) \\[4pt]
\textbf{Discard} & \text{if } \alpha(c_i) < \tau_{\alpha}^{low} \text{ or } \mu(c_i) < \tau_{\mu}^{low} \\[4pt]
\textbf{Rerank} & \text{otherwise}
\end{cases}
$$

**Threshold semantics:**
- $\tau_\alpha$: High-confidence aspect coverage gate for bypass (e.g., 0.7).
- $\tau_\mu$: High-confidence mass gate for bypass (e.g., 0.3).
- $\tau_\phi$: High-confidence focus gate for bypass (e.g., 0.8).
- $\tau_\alpha^{low}$: Below this aspect coverage, discard outright (e.g., 0.1).
- $\tau_\mu^{low}$: Below this mass, discard outright (e.g., 0.02).

**Soft-OR Variant (Single Score for Ranking):**
When a unified scalar is needed (e.g., for ranking within the Rerank queue or for threshold sweeps), combine Mass and Focus via the probabilistic noisy-OR:

$$\text{Score}(c_i) = \alpha(c_i) \cdot \text{softOR}\bigl(\mu(c_i),\, \phi(c_i)\bigr)$$

where $\text{softOR}(a, b) = a + b - ab \in [0, 1]$.

This saturates toward 1 when *either* input is high (expressing the OR semantics), is monotone in both arguments, and stays bounded — unlike the legacy additive form.

### Step 3: Budget-Coupled Admission (VRAM Enforcement)
Fixed absolute thresholds do not transfer across datasets (a dense 10-K saturates $\mu$; a sparse news stream never reaches it). We enforce hardware limits by coupling Bypass admission to the $N_{max}$ VRAM budget:

$$|\texttt{Bypass\_List}| = \min\Bigl(\,\bigl|\{c_i : \text{routed to Bypass}\}\bigr|,\; N_{max}\,\Bigr)$$

If more chunks qualify for Bypass than $N_{max}$ allows, rank them by Score descending and demote the excess into the `Rerank_Queue`. This dissolves the legacy "safe-mode overflow" edge case — overflow becomes a design invariant, not a failure mode.

---

## Pseudocode

```
Algorithm: CascadeRouter

Input:
  chunks          — list of {chunk_id, chunk_length, compressed_samples}
  query_aspects   — list of {name, weight}
  τ_α, τ_μ, τ_φ  — bypass thresholds
  τ_α_low, τ_μ_low — discard thresholds
  N_max           — VRAM budget (max bypass count)
  top_K           — max rerank queue size

Output:
  Bypass_List, Rerank_Queue

Procedure:
  total_query_weight ← Σ(a.weight for a in query_aspects)
  scored_chunks ← []

  for each chunk c_i in chunks:
    samples ← c_i.compressed_samples
    if samples is empty:
      continue  // No matches → implicitly discarded

    // --- Weight Aggregation ---
    // Each sample has max_keyword_weight (= max over contributing anchors)
    // and aspects (= dict of {aspect_name: aspect_weight})

    // --- Compute Mass (μ) ---
    sum_weighted_len ← 0
    max_weighted_len ← 0
    aspects_found ← {}

    for each sample m in samples:
      w̄ ← m.max_keyword_weight
      wl ← m.length × w̄
      sum_weighted_len ← sum_weighted_len + wl
      max_weighted_len ← max(max_weighted_len, wl)
      for each (asp_name, asp_weight) in m.aspects:
        aspects_found[asp_name] ← asp_weight  // dedup by aspect name

    μ ← sum_weighted_len / c_i.chunk_length

    // --- Compute Focus (φ) ---
    if sum_weighted_len > 0:
      φ ← max_weighted_len / sum_weighted_len
    else:
      φ ← 0

    // --- Compute Aspect Coverage (α) ---
    α ← Σ(aspects_found.values()) / total_query_weight

    // --- Compute Score (Soft-OR) ---
    score ← α × (μ + φ − μ × φ)

    // --- Three-Way Triage ---
    if α ≥ τ_α AND (μ ≥ τ_μ OR φ ≥ τ_φ):
      decision ← BYPASS
    else if α < τ_α_low OR μ < τ_μ_low:
      decision ← DISCARD
    else:
      decision ← RERANK

    scored_chunks.append({c_i, score, decision})

  // --- Budget-Coupled Admission ---
  bypass_candidates ← [c for c in scored_chunks if c.decision = BYPASS]
  rerank_candidates ← [c for c in scored_chunks if c.decision = RERANK]

  // Sort bypass candidates by score descending
  sort bypass_candidates by score DESC

  if |bypass_candidates| > N_max:
    // Demote excess to rerank
    Bypass_List  ← bypass_candidates[:N_max]
    rerank_candidates ← rerank_candidates + bypass_candidates[N_max:]
  else:
    Bypass_List ← bypass_candidates

  // Sort rerank candidates by score descending and take top-K
  sort rerank_candidates by score DESC
  Rerank_Queue ← rerank_candidates[:top_K]

  return Bypass_List, Rerank_Queue
```

---

## Conceptual Advantages
1. **Mathematical Honesty:** The legacy formula had $\rho_{cont} \le \rho_{scat}$ always, collapsing the score to $\rho_{scat}(1 + \phi)$ — a single magnitude axis scaled by a bonus in $[1, 2]$. Separating Mass ($\mu$, magnitude) from Focus ($\phi$, shape) gives the router genuine 2D expressivity.
2. **LLM Friendliness:** Bypassing on the Focus branch ($\phi \ge \tau_\phi$) isolates chunks where one long, contiguous span dominates. These are the *safest* inputs for LLM generation (no sequence-break penalty), making the LLM reranker the least valuable for exactly these chunks.
3. **Overflow Elimination:** Budget-coupled admission makes `|Bypass_List| ≤ N_max` a design invariant, removing the legacy safe-mode overflow special-casing entirely.
4. **Citable Framing:** Aligns the CPU router with established Early-Exit / Cascade Ranking literature (Wang et al., SIGIR 2011; Cambazoglu et al., WSDM 2010).
