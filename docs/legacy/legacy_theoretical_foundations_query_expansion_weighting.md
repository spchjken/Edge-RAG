> ⚠️ **LEGACY DOCUMENT** — archived for historical reference. Superseded by [`theoretical_foundations_anchored_expansion.md`](theoretical_foundations_anchored_expansion.md).
>
> **Survival map:** the IT-MPE Theorem survives there as **Theorem 1**, generalized — it now holds for any allocation distribution $p$ (Softmax is one instance), any positive IDF function, and any $K \ge 1$ (§5). Corollary 1 (high-IDF noise immunity) survives (§6.1); the RM3 derivation survives as Axiom 1 (§4.1). The adaptive similarity gate $\tau_{sim}$ and the query-dependent ceiling $\mu(Q)$ are operational policies and moved to the V7 architectural plan.

---

# 🏛️ Theoretical Foundations of Query Expansion Weighting: The Information-Theoretic Mass-Preserving Expansion (IT-MPE) Theorem

---

## 1. Problem Formulation & Overall Target

### 1.1 The Dual Objective of Lexical-Semantic Retrieval in Edge-RAG
In modern Information Retrieval (IR) and retrieval-augmented generation (RAG), search paradigms face a fundamental tension between **vocabulary mismatch** and **query drift**:
1. **Lucene BM25 (Hard Lexical Span):** Strictly constrains the search space to posting lists of explicit query terms ($\bigcup_{a \in Q} \text{PostingList}(a)$). This yields zero query drift and pristine precision on exact entities, but fails when users use conversational phrasing or synonyms.
2. **Dense & Learned Sparse Retrievers (Dense BGE & SPLADE-v3):** Project queries into continuous embedding spaces or diffuse 30k-subword vocabularies. While resolving vocabulary mismatch, they introduce varying tiers of **Query Drift**—unconstrained semantic averaging (Dense) or multi-subword accumulation (SPLADE) allows distractor documents with weak contextual associations to outvote documents containing the exact entity.

**The Edge-RAG Mandate:** In resource-constrained edge deployments, where heavy neural bi-encoders (SPLADE-v3: $174\text{s}$ TTI, $4.2\text{ GB}$ VRAM; Dense BGE-Large: $17.5\text{s}$ TTI, $2.6\text{ GB}$ VRAM) cannot be executed, Edge-RAG introduces **Anchored Lexical-Semantic Query Expansion**. The objective is to bridge vocabulary mismatch by selectively projecting corpus vocabulary terms while mathematically preserving BM25's exact discriminative boundary under strict latency ($<20\text{ms}$) and near-zero VRAM ($0.09\text{ GB}$) constraints.

### 1.2 The Scoring-Space Query Drift Dilemma
Let $Q_{\text{orig}}$ be the user's raw query, $a \in Q_{\text{orig}}$ be an anchor keyword, and $\text{Syn}(a) = \{s_1, \dots, s_K\}$ be the candidate synonyms expanded from $a$ via dense vocabulary projection ($s \sim \text{BGE}(a)$).

In Lucene BM25, the relevance score of a document $D$ for query $Q$ with term weight vector $\mathbf{w} \in \mathbb{R}^{|V|}$ is:

$$\text{Score}(D, Q) = \sum_{t \in Q} w(t) \cdot \text{IDF}(t) \cdot \psi(\text{TF}(t, D), |D|)$$

Where $\psi(\text{TF}(t, D), |D|) = \frac{\text{TF}(t, D) \cdot (k_1 + 1)}{\text{TF}(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgDL}}\right)}$ is the BM25 term frequency saturation factor.

#### The Core Vulnerability:
If query expansion constrains **only** the query-side term weight vector ($w(s) \le \mu \cdot w(a)$), it does **NOT** guarantee stability in the actual BM25 score space, because $w(s)$ is multiplied by **$\text{IDF}(s)$**.

When an expanded candidate synonym $s$ is rare across the corpus ($\text{IDF}(s) \gg \text{IDF}(a)$), a distractor document $D_{\text{distractor}}$ containing a single mention of $s$ can achieve a BM25 score higher than a relevant document $D_{\text{gold}}$ containing the exact user anchor $a$:

$$\text{Score}(s, D_{\text{distractor}}) = w(s) \cdot \text{IDF}(s) \cdot \psi(1) \gg w(a) \cdot \text{IDF}(a) \cdot \psi(1) = \text{Score}(a, D_{\text{gold}})$$

*Concrete Empirical Example:* In biomedical retrieval, anchor $a = \text{"EHR"}$ ($\text{IDF} \approx 2.5$) expanded into rare cancer gene $s = \text{"erbb2"}$ ($\text{IDF} \approx 7.5$). Even with a conservative query weight $w(s) = 0.33 \cdot w(a)$, the product $w(s) \cdot \text{IDF}(s) = 0.33 \times 7.5 = 2.48 \approx w(a) \cdot \text{IDF}(a)$. Distractor papers mentioning `erbb2` tied or outranked true gold papers talking about EHR, dropping the gold chunk from **Rank 1 to Rank 18**.

### 1.3 The Overall Target
We require a mathematically grounded weighting function that strictly guarantees **Score-Space Anchor Dominance**: for any anchor $a$ and any document $D$, the total BM25 scoring mass of all expanded synonyms for $a$ cannot exceed a fraction $\mu \le 0.35$ of the anchor's own scoring mass:

$$\frac{\sum_{s \in \text{Syn}(a)} \text{Score}(s, D)}{\text{Score}(a, D^*)} \le \mu \le 0.35$$

---

## 2. Baseline Theoretical Principles

```text
                      ┌──────────────────────────────────────────────┐
                      │             User Query Intent Q              │
                      └──────────────────────┬───────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
          [Primary Anchor Mass]                         [Latent Expansion Mass]
       Explicit User Prior: P=1.0                      Probabilistic Association: P(s|a)
     w(a) = w_base · w_POS · (1 + γ·IDF)               w(s) = μ(Q) · w(a) · min(1, IDF_a/IDF_s) · Softmax
                     │                                               │
                     └───────────────────────┬───────────────────────┘
                                             │
                                             ▼
                     ┌──────────────────────────────────────────────┐
                     │    Mass Invariant: Σ w(s) ≤ μ(Q) · w(a)      │
                     │         with Hard Ceiling μ(Q) ≤ 0.35        │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
                     ┌──────────────────────────────────────────────┐
                     │    Score-Space Guarantee:                    │
                     │    Score(Syn(a)) ≤ 0.35 · Score(Anchor a)     │
                     └──────────────────────────────────────────────┘
```

### 2.1 Relevance Model 3 (RM3) Query Prior Interpolation
In probabilistic language modeling and pseudo-relevance feedback (Lavrenko & Croft, 2001; Abdul-Jaleel et al., 2004), the expanded query language model $P(t \mid Q')$ is formed via linear interpolation between the maximum-likelihood user intent $P(t \mid Q_{\text{orig}})$ and the latent expansion model $P(t \mid \theta_{\text{expansion}})$:

$$P(t \mid Q') = \lambda P(t \mid Q_{\text{orig}}) + (1 - \lambda) P(t \mid \theta_{\text{expansion}})$$

Across extensive empirical consensus on TREC and web benchmarks, the optimal query conservatism prior is bounded in the range:

$$\lambda^* \in [0.70, \ 0.80] \quad \text{with canonical peak at } \lambda^* \approx 0.75$$

The resulting expansion-to-anchor mass ratio ceiling is:

$$\mu_{\text{max}} = \frac{1 - \lambda^*}{\lambda^*} = \frac{1 - 0.75}{0.75} = \frac{1}{3} \approx 0.35$$

When $\mu > 0.35$ ($\lambda < 0.74$), the expansion mass starts overpowering the explicit anchor constraints, causing severe query drift.

### 2.2 Bayesian Prior vs. Conditional Semantic Association
In Edge-RAG's dense projection framework:
1. **Anchor Terms ($a \in Q_{\text{orig}}$):** Represent observed, unambiguous user constraints. The prior probability of intent relevance is $P(\text{Intent} \mid a) = 1.0$.
2. **Expansion Terms ($s \notin Q_{\text{orig}}$):** Represent unobserved semantic hypotheses inferred via dense embedding projection. Its relevance probability is strictly conditional:
   $$P(\text{Intent} \mid s) = P(s \mid a) \cdot P(\text{Intent} \mid a) \approx \text{CosSim}(\mathbf{e}_s, \mathbf{e}_a) \cdot P(\text{Intent} \mid a)$$

---

## 3. Proposed Formula: IDF-Damped Continuous IT-MPE

To eliminate quantization noise from discrete integer repetition while guaranteeing score-space mass preservation, Edge-RAG Pipeline V7 (`BM25Dense_V7`) computes continuous sparse query weight vectors $\mathbf{w} \in \mathbb{R}^{|V|}$ through three unified equations:

### 3.1 Anchor Base Weighting ($w(a)$)
For each non-stopword token $a \in Q_{\text{orig}}$:

$$w(a) = \text{Weight}_{\text{POS}}(a) \times \left(1.0 + \gamma \cdot \frac{\text{IDF}(a)}{\max_{t \in Q} \text{IDF}(t)} \cdot \text{Centrality}(a)\right)$$

Where:
* $\text{Weight}_{\text{POS}}(\text{Noun / Entity}) = 1.25, \quad \text{Weight}_{\text{POS}}(\text{Verb}) = 0.85, \quad \text{Weight}_{\text{POS}}(\text{Adjective / Other}) = 0.70$.
* $\text{Centrality}(a) = \frac{1}{|Q|-1} \sum_{t \in Q \setminus \{a\}} \cos(\mathbf{e}_a, \mathbf{e}_t)$ measures semantic cohesion within the query.
* $\gamma = 2.0 \implies w(a) \in [1.0, \ 3.75]$.

### 3.2 Adaptive Similarity Gating ($\tau_{\text{sim}}(a)$)
To protect high-information entities from diffuse semantic expansion while allowing umbrella terms (`"qwen"`, `"EHR"`, `"RAG"`) to reach their specific variants, the similarity threshold dynamically scales with anchor IDF:

$$\tau_{\text{sim}}(a) = \tau_{\text{base}} + \Delta \tau \cdot \left(\frac{\text{IDF}(a)}{\text{IDF}_{\text{max\_corpus}}}\right) = 0.80 + 0.10 \times \left(\frac{\text{IDF}(a)}{\text{IDF}_{\text{max}}}\right) \quad \implies \tau_{\text{sim}} \in [0.80, \ \mathbf{0.90}]$$

### 3.3 IDF-Damped Softmax Synonym Weight Allocation ($w(s_k \mid a)$)
For candidate terms $s_k \in \text{Syn}(a)$ satisfying $\text{CosSim}(\mathbf{e}_{s_k}, \mathbf{e}_a) \ge \tau_{\text{sim}}(a)$:

$$w(s_k \mid a) = \mu(Q) \cdot w(a) \cdot \mathbf{\min\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right)} \cdot \frac{\exp\left(\frac{\text{CosSim}(\mathbf{e}_{s_k}, \mathbf{e}_a)}{\tau}\right)}{\sum_{j=1}^K \exp\left(\frac{\text{CosSim}(\mathbf{e}_{s_j}, \mathbf{e}_a)}{\tau}\right)}$$

Where:
* $\mu(Q) = 0.35 \times \left(1 - 0.5 \times \frac{\max_{t \in Q} \text{IDF}(t)}{\text{IDF}_{\text{max\_corpus}}}\right) \le \mathbf{0.35}$ (RM3 Mass Ceiling).
* $\min\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right)$ is the **IDF Damping Factor**.
* $\tau = 0.10$ is the **Softmax Temperature**, providing aggressive exponential attenuation for weaker candidates ($\text{CosSim} < 0.85$).
* $K = C_{\text{exp}} \le 2$ (Capacity cap).

---

## 4. Mathematical Proof: Score-Space Anchor Dominance

**Theorem 1 (Information-Theoretic Mass-Preserving Expansion Theorem):**
Let $a \in Q_{\text{orig}}$ be an anchor keyword with weight $w(a)$, and let $\text{Syn}(a) = \{s_1, \dots, s_K\}$ be its candidate synonyms weighted according to Equation (3.3). 

For any gold document $D_{\text{gold}}$ matching anchor $a$ with frequency $\text{TF}(a, D_{\text{gold}}) \ge 1$ and any distractor document $D_{\text{distractor}}$ matching all $K$ expanded synonyms $\{s_1, \dots, s_K\}$ with frequency $\text{TF}(s_k, D_{\text{distractor}}) = 1$ and zero occurrences of $a$, under normalized document lengths ($|D_{\text{gold}}| \approx |D_{\text{distractor}}| \approx \text{avgDL}$), the aggregate BM25 score of all expanded synonyms is strictly bounded by:

$$\sum_{k=1}^K \text{Score}(s_k, D_{\text{distractor}}) \le \mu(Q) \cdot \text{Score}(a, D_{\text{gold}}) \le \mathbf{0.35 \times \text{Score}(a, D_{\text{gold}})}$$

---

### Proof:

#### Step 1: Formulate the Individual Document Scores in BM25
Under normalized document length ($|D| = \text{avgDL}$), the BM25 TF saturation term for a single occurrence ($\text{TF}=1$) simplifies to:

$$\psi(1) = \frac{1 \cdot (k_1 + 1)}{1 + k_1 \cdot (1 - b + b \cdot 1)} = \frac{k_1 + 1}{1 + k_1} = 1.0$$

The BM25 score of $D_{\text{gold}}$ matching anchor $a$ ($\text{TF}=1$) is:

$$\text{Score}(a, D_{\text{gold}}) = w(a) \cdot \text{IDF}(a) \cdot \psi(1) = w(a) \cdot \text{IDF}(a)$$

The aggregate BM25 score of $D_{\text{distractor}}$ matching all $K$ synonyms $s_1, \dots, s_K$ ($\text{TF}=1$ for each) is:

$$\text{Score}(\text{Syn}(a), D_{\text{distractor}}) = \sum_{k=1}^K w(s_k \mid a) \cdot \text{IDF}(s_k) \cdot \psi(1) = \sum_{k=1}^K w(s_k \mid a) \cdot \text{IDF}(s_k)$$

---

#### Step 2: Substitute the Proposed Synonym Weight Formula
Substitute $w(s_k \mid a)$ from Equation (3.3) into the score summation:

$$\text{Score}(\text{Syn}(a), D_{\text{distractor}}) = \sum_{k=1}^K \left[ \mu(Q) \cdot w(a) \cdot \min\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right) \cdot \sigma_k \right] \cdot \text{IDF}(s_k)$$

Where $\sigma_k = \frac{\exp(\text{CosSim}(\mathbf{e}_{s_k}, \mathbf{e}_a) / \tau)}{\sum_{j=1}^K \exp(\text{CosSim}(\mathbf{e}_{s_j}, \mathbf{e}_a) / \tau)}$ is the Softmax probability distribution over the $K$ candidates, satisfying $\sum_{k=1}^K \sigma_k = 1.0$.

---

#### Step 3: Evaluate the Product Under Both IDF Regimes

##### Case A: Synonym is Rare ($\text{IDF}(s_k) \ge \text{IDF}(a)$)
When candidate synonym $s_k$ has higher IDF than anchor $a$:

$$\min\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right) = \frac{\text{IDF}(a)}{\text{IDF}(s_k)}$$

Multiplying by $\text{IDF}(s_k)$:

$$\left[\frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right] \cdot \text{IDF}(s_k) = \text{IDF}(a)$$

The rare synonym's inflated $\text{IDF}(s_k)$ is **identically cancelled out**.

##### Case B: Synonym is Common ($\text{IDF}(s_k) < \text{IDF}(a)$)
When candidate synonym $s_k$ is more common than anchor $a$:

$$\min\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right) = 1.0$$

Multiplying by $\text{IDF}(s_k)$:

$$1.0 \cdot \text{IDF}(s_k) = \text{IDF}(s_k) < \text{IDF}(a)$$

---

#### Step 4: Aggregate Over All $K$ Candidates
Combining Case A and Case B, for every candidate $k \in \{1, \dots, K\}$:

$$\min\left(1.0, \ \frac{\text{IDF}(a)}{\text{IDF}(s_k)}\right) \cdot \text{IDF}(s_k) \le \text{IDF}(a)$$

Factoring out the invariant terms from the summation:

$$\text{Score}(\text{Syn}(a), D_{\text{distractor}}) \le \sum_{k=1}^K \left[ \mu(Q) \cdot w(a) \cdot \text{IDF}(a) \cdot \sigma_k \right]$$

$$\text{Score}(\text{Syn}(a), D_{\text{distractor}}) \le \mu(Q) \cdot w(a) \cdot \text{IDF}(a) \cdot \underbrace{\sum_{k=1}^K \sigma_k}_{= 1.0}$$

$$\text{Score}(\text{Syn}(a), D_{\text{distractor}}) \le \mu(Q) \cdot \left[ w(a) \cdot \text{IDF}(a) \right]$$

Recognizing that $\text{Score}(a, D_{\text{gold}}) = w(a) \cdot \text{IDF}(a)$:

$$\mathbf{\text{Score}(\text{Syn}(a), D_{\text{distractor}}) \le \mu(Q) \cdot \text{Score}(a, D_{\text{gold}})}$$

Since $\mu(Q) \le \mu_{\text{max}} = 0.35$:

$$\mathbf{\frac{\text{Score}(\text{Syn}(a), D_{\text{distractor}})}{\text{Score}(a, D_{\text{gold}})} \le 0.35}$$

$\blacksquare$

---

### 4.2 Corollary: Resistance to Ultra-Rare Out-of-Domain Noise
**Corollary 1 (Immunity to High-IDF Drift):**
Let $s_{\text{noise}}$ be an arbitrary out-of-domain token or rare singleton with arbitrarily high IDF ($\text{IDF}(s_{\text{noise}}) \to \infty$). 

Under Equation (3.3), its individual scoring impact on any document is strictly bounded by:

$$\text{Score}(s_{\text{noise}}, D) \le \mu(Q) \cdot w(a) \cdot \text{IDF}(a) \cdot \sigma_{\text{noise}} \le 0.35 \cdot \text{Score}(a, D_{\text{gold}})$$

Even if an out-of-domain term passes initial cosine filtering, it can never contribute more than $35\%$ of the anchor's score, completely eliminating catastrophic query drift.

---

## 5. Summary Comparison Matrix

| Retrieval Paradigm | Weight Representation | High-IDF Synonym Handling | Score-Space Mass Guarantee | Theoretical Foundation |
| :--- | :---: | :---: | :---: | :--- |
| **Naive Expansion ($R=1$)** | Unweighted ($w=1$) | Unchecked ($\text{IDF}_s$ dominates) | ❌ None (Score ratio up to $5.0\times$) | None (Heuristic) |
| **Integer Repetition (Schemas 1–6b)** | Discrete $R \in [2, 5]$ | $R_s \ge 1$ (Cannot scale down rare words) | ❌ Broken when $\text{IDF}_s > \text{IDF}_a$ | Discrete RM3 Proxy |
| **Proposed Unified V7 (`BM25Dense_V7`)** | **Continuous $\mathbf{w} \in \mathbb{R}^{|V|}$** | **$\min\left(1.0, \frac{\text{IDF}_a}{\text{IDF}_s}\right)$ Damped** | **✅ Mathematically $\le 0.35$** | **RM3 + Score-Space IT-MPE** |
