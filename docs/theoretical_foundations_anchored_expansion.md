# 🏛️ Theoretical Foundations of Anchored Lexical-Semantic Query Expansion: The Information-Theoretic Mass-Preserving Expansion (IT-MPE) Theorem

---

## 1. Epistemic Status & Scope

### 1.1 Epistemic Hierarchy Matrix

| Component | Classification | Status | Foundation |
| :--- | :--- | :--- | :--- |
| **Score-Space Anchor Dominance (Theorem 1, IT-MPE)** | Theorem | ✅ Proven | Exact algebraic bound in the BM25/Lucene score space; assumptions explicit (§5.1) |
| **High-IDF Noise Immunity (Corollary 1)** | Corollary | ✅ Proven | Follows from Theorem 1 with $\text{IDF}(s) \to \infty$ |
| **Query-Level Bound (Corollary 2)** | Corollary | ✅ Proven | Summation of Theorem 1 over anchors; justifies global capacity budgets |
| **Expansion Mass Ceiling $\mu$ (Axiom 1)** | Axiom | ⚙️ Free parameter | $\mu \in (0, 1]$ — the ceiling on expansion mass relative to anchor mass; a calibration choice, not a theorem value (§4.1) |
| **Anchor weight composition, candidate generation, gating, capacity policy** | Given inputs / operational parameters | ⚙️ Out of scope | Specified by the V7 architectural plan |

### 1.2 Scope & Consolidation Note

This document establishes the **weighting law** for anchored lexical-semantic query expansion and proves the resulting score-space drift bound. It deliberately does **not** cover:

- how anchors are weighted internally (centrality, POS priors, $\gamma$),
- how candidate terms are generated (dense vocabulary projection, inclusive vocabulary rescue),
- how candidates are filtered (similarity gating $\tau_{sim}$),
- how many terms are expanded per anchor or per query (capacity budgets).

These are operational parameters of the retrieval architecture, not part of the theorem; they are specified in the V7 architectural plan.

*Consolidation:* this document supersedes [`legacy_theoretical_foundations_query_expansion_weighting.md`](legacy_theoretical_foundations_query_expansion_weighting.md) and [`legacy_theoretical_foundations_expansion_capacity.md`](legacy_theoretical_foundations_expansion_capacity.md). The former *"Theorem 1 (Information-Theoretic Optimality of Saliency-Proportional Capacity)"* from the capacity document is **retired**: its optimality claim rested on unverified assumptions (a linear candidate-utility model $\rho(a) = \alpha \cdot I(a)$ and an unmeasurable noise-entropy rate $\epsilon$). Capacity is treated here purely as an operational parameter (§5.3, Remark 2).

### 1.3 Core Invariant

For every anchor $a$ and every expansion set $\text{Syn}(a)$, the total BM25 scoring mass of the expansion terms cannot exceed a fraction $\mu$ of the anchor's own scoring mass:

$$\frac{\sum_{s \in \text{Syn}(a)} \text{Score}(s, D)}{\text{Score}(a, D^*)} \le \mu$$

under the explicit conditions of Theorem 1, where $\mu \in (0, 1]$ is the expansion mass ceiling (Axiom 1).

---

## 2. Problem Formulation: The Scoring-Space Drift Dilemma

### 2.1 Vocabulary Mismatch vs. Query Drift

Retrieval systems face a fundamental tension:

1. **Lexical scoring (Lucene BM25)** strictly constrains the search to posting lists of explicit query terms. This yields zero query drift and pristine precision on exact entities, but fails when users employ conversational phrasing or synonyms (vocabulary mismatch).
2. **Dense scoring** projects queries and documents into continuous embedding spaces, resolving vocabulary mismatch but introducing **query drift**: unconstrained semantic averaging lets documents with weak contextual associations outvote documents containing the exact entity.

**Anchored lexical-semantic query expansion** bridges the two: vocabulary terms semantically close to each anchor are projected into the query, but with weights that preserve the lexical scorer's discriminative boundary. The central theoretical problem is to choose those weights so that expansion cannot overpower the anchors it extends.

### 2.2 Score-Space Information Mass

The retriever scores documents in BM25/Lucene space. For a query term set $Q$ with weight vector $\mathbf{w}$:

$$\text{Score}(D, Q) = \sum_{t \in Q} w(t) \cdot \text{IDF}(t) \cdot \psi(\text{TF}(t, D), |D|) \tag{1}$$

$$\psi(\text{TF}, |D|) = \frac{\text{TF} \cdot (k_1 + 1)}{\text{TF} + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgDL}}\right)} \tag{2}$$

Define the **score-space information mass** of term $t$:

$$M(t) = w(t) \cdot \text{IDF}(t) \tag{3}$$

the per-occurrence score contribution of $t$. A document's score for a term set is $\sum_t M(t) \cdot \psi(\cdot)$. Query drift is therefore a statement about **mass in score space**, not about query weights alone.

### 2.3 The Core Vulnerability: High-IDF Synonym Hijacking

A query-side-only bound on weights — e.g., $w(s) \le \mu \cdot w(a)$ — does **not** guarantee stability in score space, because $w(s)$ is multiplied by $\text{IDF}(s)$. When a candidate synonym $s$ is rare across the corpus ($\text{IDF}(s) \gg \text{IDF}(a)$), a distractor document $D_{dist}$ containing a single mention of $s$ can approach or exceed the score of a gold document $D_{gold}$ containing the exact anchor:

$$M(s) = w(s) \cdot \text{IDF}(s) \quad \text{vs.} \quad M(a) = w(a) \cdot \text{IDF}(a)$$

*Concrete example:* in biomedical retrieval, anchor $a = \text{"EHR"}$ ($\text{IDF} \approx 2.5$) expanded into rare cancer gene $s = \text{"erbb2"}$ ($\text{IDF} \approx 7.5$). Even with a query weight $w(s) = 0.5 \cdot w(a)$ (half the anchor's weight), the mass $M(s) = 0.5 \times 7.5 = 3.75 > M(a) = 2.5$. Distractor papers mentioning `erbb2` outranked true gold papers about EHR — the gold chunk dropped from **Rank 1 to Rank 18**. With $K$ synonyms accumulating, the aggregate can exceed the anchor outright.

### 2.4 The Overall Target

We require a weighting function $w(s_k \mid a)$ that strictly guarantees **score-space anchor dominance** for **any number $K \ge 1$ of expansion terms**: the total BM25 scoring mass of all expanded synonyms of $a$ cannot exceed a fraction $\mu$ of the anchor's own scoring mass.

---

## 3. Notation & The IT-MPE Weighting Formula

### 3.1 Notation

| Symbol | Meaning |
| :--- | :--- |
| $Q$, $a$ | Query after stopword removal; anchor term $a \in Q$ |
| $w(a)$ | Positive BM25/Lucene query weight of $a$; its internal composition is external to this theory |
| $\text{Syn}(a) = \{s_1, \dots, s_K\}$ | Candidate expansion terms of $a$; $K \ge 1$ arbitrary |
| $\text{IDF}(t)$ | Inverse document frequency of $t$; positive for every indexed term |
| $\mu$ | Expansion mass ceiling; a free parameter $\mu \in (0, 1]$ (Axiom 1) |
| $p = (p_1, \dots, p_K)$ | Allocation distribution over $\text{Syn}(a)$: $p_k \ge 0$, $\sum_{k=1}^K p_k = 1$ |
| $\psi(\text{TF}, |D|)$ | BM25 term-frequency saturation factor, Eq. (2) |
| $M(t) = w(t) \cdot \text{IDF}(t)$ | Score-space information mass of term $t$, Eq. (3) |
| $D_{gold}$, $D_{dist}$ | Gold document matching the anchor; distractor document matching only synonyms |

### 3.2 Given Inputs (Not Part of the Theorem)

The theorem treats as given, and does not prove anything about:

- **Anchor weights** $w(a) > 0$ — however composed (centrality, POS priors, IDF scaling);
- **Candidate sets** $\text{Syn}(a)$ of arbitrary size $K \ge 1$ — however generated (dense vocabulary projection, inclusive vocabulary rescue);
- **Inverse document frequencies** — any positive IDF weighting, including Lucene's exact form.

### 3.3 The IT-MPE Weighting Formula

For candidate terms $s_k \in \text{Syn}(a)$:

$$w(s_k \mid a) = \mu \cdot w(a) \cdot \min\!\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right) \cdot p_k \tag{4}$$

Three immediate observations:

1. **Weight-space mass invariant:** $\sum_k w(s_k \mid a) \le \mu \cdot w(a)$, since $\min(\cdot) \le 1$ and $\sum_k p_k = 1$. This alone is *necessary but not sufficient* (§2.3); the damping factor is what transfers the bound into score space.
2. **Any allocation distribution works:** the formula requires only that $p$ is a probability distribution. The Softmax over candidate similarities (with a temperature constant) is *one admissible instance*; the theorem does not depend on it.
3. **Continuity is necessary:** discrete integer repetition ($w(s) = R \in \mathbb{Z}$, $R \ge 1$) cannot represent a damping factor below $1$. When $\text{IDF}(s) > \text{IDF}(a)$, the required factor $\text{IDF}(a)/\text{IDF}(s) < 1$ is unrepresentable, so score-space dominance is unattainable. Continuous weights $\mathbf{w} \in \mathbb{R}^{|V|}$ are required.

---

## 4. Baseline: The Expansion Mass Ceiling (Axiom 1)

### 4.1 Relevance Model 3 (RM3) Query Prior Interpolation

The relevance-model framework is due to Lavrenko & Croft (2001, *Relevance-Based Language Models*, SIGIR); the **RM3** "query mix" that interpolates the relevance model with the original query is due to Abdul-Jaleel et al. (2004, *UMass at TREC 2004*). In RM3, the expanded query language model is a linear interpolation between the original query and the latent expansion model:

$$P(t \mid Q') = \lambda \, P(t \mid Q_{orig}) + (1 - \lambda) \, P(t \mid \theta_{expansion})$$

The interpolation weight $\lambda$ is an **operational setting**. Common values span $\lambda \in [0.5, 0.8]$, with $\lambda = 0.5$ the common toolkit default; reported optima vary by corpus and query. The expansion-to-anchor mass ratio is a function of the chosen $\lambda$:

$$\mu(\lambda) = \frac{1 - \lambda}{\lambda}$$

| $\lambda$ | $\mu = (1-\lambda)/\lambda$ |
| :---: | :---: |
| 0.50 (toolkit default) | 1.00 |
| 0.60 | 0.67 |
| 0.70 | 0.43 |
| 0.75 | 0.33 |
| 0.80 | 0.25 |

**Axiom 1 (Expansion Mass Ceiling):** for every anchor $a$,
$\sum_{k=1}^{K} w(s_k \mid a) \le \mu \cdot w(a)$, where $\mu \in (0, 1]$ is a free parameter — the ceiling on expansion mass relative to anchor mass. The value of $\mu$ is a calibration choice external to this theory.

### 4.2 Bayesian Prior vs. Conditional Semantic Association

1. **Anchor terms ($a \in Q_{orig}$):** observed, unambiguous user constraints; the prior probability of intent relevance is $P(\text{Intent} \mid a) = 1.0$.
2. **Expansion terms ($s \notin Q_{orig}$):** unobserved semantic hypotheses; their relevance is strictly conditional:
   $$P(\text{Intent} \mid s) = P(s \mid a) \cdot P(\text{Intent} \mid a) \approx \text{CosSim}(\mathbf{e}_s, \mathbf{e}_a) \cdot 1.0$$

The allocation distribution $p$ of Eq. (4) instantiates the relative conditional association $P(s_k \mid a)$ among the admitted candidates.

---

## 5. Theorem 1 (IT-MPE) & Proof

### 5.1 Statement

**Theorem 1 (Information-Theoretic Mass-Preserving Expansion).**
Let $a$ be an anchor with query weight $w(a) > 0$, and let $\text{Syn}(a) = \{s_1, \dots, s_K\}$ be any finite candidate set with $\text{IDF}(s_k) > 0$ for all $k$, weighted according to Eq. (4) with any $\mu > 0$. Consider two documents with normalized lengths $|D_{gold}| \approx |D_{dist}| \approx \text{avgDL}$, where

- $D_{gold}$ matches anchor $a$ with frequency $\text{TF}(a, D_{gold}) \ge 1$,
- $D_{dist}$ matches each synonym with frequency $\text{TF}(s_k, D_{dist}) = 1$ and the anchor with $\text{TF}(a, D_{dist}) = 0$.

Then the aggregate BM25 score of all expanded synonyms is strictly bounded by:

$$\sum_{k=1}^{K} \text{Score}(s_k, D_{dist}) \le \mu \cdot \text{Score}(a, D_{gold})$$

### 5.2 Proof

#### Step 1: The BM25 saturation factor under the theorem's conditions
For $|D| = \text{avgDL}$, Eq. (2) gives $\psi(1) = (k_1 + 1)/(1 + k_1) = 1$. Moreover $\psi$ is strictly increasing in TF:

$$\frac{\partial \psi}{\partial \text{TF}} = \frac{(k_1 + 1) \cdot k_1 \cdot \left(1 - b + b \frac{|D|}{\text{avgDL}}\right)}{\left(\text{TF} + k_1 \left(1 - b + b \frac{|D|}{\text{avgDL}}\right)\right)^2} > 0$$

so $\text{TF}(a, D_{gold}) \ge 1$ implies:

$$\text{Score}(a, D_{gold}) = w(a) \cdot \text{IDF}(a) \cdot \psi(\text{TF}_a) \ge w(a) \cdot \text{IDF}(a) \cdot \psi(1) = w(a) \cdot \text{IDF}(a)$$

The distractor's aggregate synonym score is:

$$\text{Score}(\text{Syn}(a), D_{dist}) = \sum_{k=1}^{K} w(s_k \mid a) \cdot \text{IDF}(s_k) \cdot \psi(1) = \sum_{k=1}^{K} w(s_k \mid a) \cdot \text{IDF}(s_k)$$

#### Step 2: Substitute the weighting formula
$$\text{Score}(\text{Syn}(a), D_{dist}) = \mu \cdot w(a) \cdot \sum_{k=1}^{K} \left[ \min\!\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right) \cdot \text{IDF}(s_k) \right] \cdot p_k$$

#### Step 3: Evaluate the damping product under both IDF regimes

- **Case A — rare synonym ($\text{IDF}(s_k) \ge \text{IDF}(a)$):**
  $\min(1, \text{IDF}_a/\text{IDF}_s) = \text{IDF}_a/\text{IDF}_s$, so the product $= \text{IDF}(a)$. The rare synonym's inflated IDF is **identically cancelled**.
- **Case B — common synonym ($\text{IDF}(s_k) < \text{IDF}(a)$):**
  $\min(1, \text{IDF}_a/\text{IDF}_s) = 1$, so the product $= \text{IDF}(s_k) < \text{IDF}(a)$.

In both cases: $\min(1, \text{IDF}_a/\text{IDF}_{s_k}) \cdot \text{IDF}(s_k) \le \text{IDF}(a)$.

#### Step 4: Aggregate over the allocation distribution
$$\text{Score}(\text{Syn}(a), D_{dist}) \le \mu \cdot w(a) \cdot \text{IDF}(a) \cdot \underbrace{\sum_{k=1}^{K} p_k}_{= 1.0} \le \mu \cdot \left[ w(a) \cdot \text{IDF}(a) \right] \le \mu \cdot \text{Score}(a, D_{gold})$$

so $\text{Score}(\text{Syn}(a), D_{dist}) \le \mu \cdot \text{Score}(a, D_{gold})$. $\blacksquare$

### 5.3 Remarks

**Remark 1 (General TF / multi-mention).** Without the single-mention assumption, the bound generalizes to:

$$\text{Score}(\text{Syn}(a), D') \le \mu \cdot \text{Score}(a, D^{*}) \cdot \frac{\max_k \psi(\text{TF}(s_k, D'))}{\psi(\text{TF}(a, D^{*}))}$$

Since $\psi(\text{TF}) > 1$ for $\text{TF} \ge 2$ (at $|D| = \text{avgDL}$), the *proven* guarantee is the single-mention worst case that defines the hijack pathology of §2.3; multi-mention accumulation of many synonyms is the operational motivation for capacity budgets and candidate filtering, addressed by the architectural plan.

**Remark 2 (K-invariance).** No proof step depends on $K$. The bound holds for **any candidate support size $K \ge 1$**, so expansion capacity is a purely operational parameter (coverage vs. retrieval latency) with no bearing on the mass guarantee.

**Remark 3 (Robustness).** The proof uses only the positivity of $\text{IDF}$ and $\sum_k p_k = 1$: it holds for any positive IDF weighting, including Lucene's exact form, and for any allocation distribution $p$. Lucene's query/document norms scale both sides of the score ratio by identical constants, so the bound survives normalization.

---

## 6. Corollaries

### 6.1 Corollary 1: Immunity to High-IDF Out-of-Domain Noise
Let $s_{noise}$ be an arbitrary out-of-domain token with arbitrarily high IDF ($\text{IDF}(s_{noise}) \to \infty$). Under Eq. (4), its individual scoring impact on any document is bounded by:

$$\text{Score}(s_{noise}, D) \le \mu \cdot w(a) \cdot \text{IDF}(a) \cdot p_{noise} \le \mu \cdot \text{Score}(a, D_{gold})$$

Even if an out-of-domain term is admitted as a candidate, its contribution is bounded by $\mu \cdot \text{Score}(a, D_{gold})$ — so catastrophic query drift is eliminated by construction.

### 6.2 Corollary 2: Query-Level Bound (Composition Across Anchors)
Summing Theorem 1 over all anchors $a \in Q$: for any document $D'$ matching only expanded synonyms (single mention each) and any document $D^{*}$ matching every anchor at least once, with all lengths $\approx \text{avgDL}$:

$$\text{Score}(\text{Syn}(Q), D') = \sum_{a \in Q} \text{Score}(\text{Syn}(a), D') \le \mu \cdot \sum_{a \in Q} w(a) \cdot \text{IDF}(a) \le \mu \cdot \text{Score}(Q, D^{*})$$

**Consequence:** the drift guarantee composes across anchors. Capacity budgets may therefore be allocated **globally** over the whole augmented query — any distribution of expansion slots across anchors preserves every per-anchor bound.

---

## 7. Falsification Criteria (Theory Verification Only)

This section defines how the *theorem itself* can be checked against real retrieval runs. Policy tuning (capacity budgets, gating thresholds, anchor-weight composition) is evaluated in the experimental documents, not here.

- **F1 — The bound holds empirically.** On benchmark retrievals, measure the maximum score ratio over all retrieved documents, $r_{max} = \max_D \text{Score}(\text{Syn}(Q), D) / \text{Score}(Q, D^{*})$, and verify $r_{max} \le \mu_{max}$ under the $\psi$-correction of Remark 1. Because Theorem 1 is algebraic, any violation indicates an implementation defect rather than a flaw in the theory — the check therefore doubles as a correctness oracle for the weighting code.
- **F2 — The bounded case is the relevant case.** Confirm that observed drift events (documents outranking the gold) are dominated by the single-mention, rare-synonym hijack pattern bounded by Theorem 1 — rather than by multi-mention accumulation (Remark 1) or by documents matching both anchors and synonyms. If F2 fails, the theorem is *true but insufficient*, and the drift budget must be extended beyond Theorem 1.

---

## 8. Summary Comparison Matrix

| Retrieval Paradigm | Weight Representation | Score-Space Guarantee vs. Anchor | Limitation |
| :--- | :---: | :---: | :--- |
| **Naive Expansion** | Unweighted ($w(s) = 1$ flat) | ❌ None — mass ratio up to $\text{IDF}(s)/\text{IDF}(a)$ | No damping, no mass ceiling |
| **Discrete Repetition (V2, $R \in [2, 5]$)** | Integer $R \ge 1$ | ❌ Broken when $\text{IDF}(s) > \text{IDF}(a)$ | Cannot represent damping $< 1$ (§3.3, obs. 3) |
| **Continuous IT-MPE (Eq. 4)** | Continuous $\mathbf{w} \in \mathbb{R}^{|V|}$ | ✅ $\le \mu \cdot \text{Score}(a)$ for any $K \ge 1$ ($\mu \in (0,1]$, Axiom 1) | Bounded drift only; recall and semantic fit are governed by candidate generation & filtering (§1.2) |
