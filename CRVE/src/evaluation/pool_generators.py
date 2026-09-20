"""
src/evaluation/pool_generators.py

Deterministic Vocabulary Pool Generators for Edge-RAG Stage 1 Oracle Isolation.
Implements the 5 formal pool policies with guaranteed nestedness (P_B = pi[:B]),
deterministic lexicographic tie-breaking, CELF-optimized submodular coverage,
and strictly disjoint 6-band corpus-aware DF partitioning.
"""

import math
import heapq
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Any, Callable


class PoolPolicy(str, Enum):
    SALIENCE = "salience"
    SPECIFICITY = "specificity"
    COVERAGE = "coverage"
    STRATIFIED = "stratified"
    HYBRID = "hybrid"


# Standard capacity budgets
FIXED_BUDGETS = [1000, 2500, 5000, 10000, 15000, 20000]
PERCENTAGE_CUTOFFS = [0.02, 0.05, 0.10, 0.20]
MAX_CAPACITY = 20000


def get_df_band(df: int, num_docs: int) -> str:
    """
    Returns the strictly disjoint corpus-aware DF band for a term.
    Bands 4-6 strictly require DF > 20, guaranteeing that every term
    belongs to exactly one band.
    """
    if df <= 0:
        return "DF=0"
    if df == 1:
        return "DF=1"
    if 2 <= df <= 5:
        return "DF=2-5"
    if 6 <= df <= 20:
        return "DF=6-20"
    
    # For df > 20, partition by relative collection frequency df / N
    rel_df = df / max(num_docs, 1)
    if rel_df < 0.001:  # < 0.1%
        return "DF>20_rel<0.1%"
    elif rel_df <= 0.01:  # 0.1% to 1.0%
        return "DF>20_rel_0.1-1%"
    else:  # > 1.0%
        return "DF>20_rel>1%"


def is_eligible_term(term: str, df: int, cf: int, num_docs: int) -> bool:
    """
    Operational eligibility reliability filter:
    DF >= 2, CF >= 3, len >= 2, not pure digits, and DF/N <= 0.15.
    """
    if len(term) < 2:
        return False
    if df < 2 or cf < 3:
        return False
    if (df / max(num_docs, 1)) > 0.15:
        return False
    if term.isdigit():
        return False
    return True


def is_index_compound_or_alphanumeric(term: str) -> bool:
    """
    Index-visible compound or technical alphanumeric entity recognizer.
    Returns True if term contains digits or non-alphanumeric punctuation.
    """
    has_digit = any(c.isdigit() for c in term)
    has_punct = any(not c.isalnum() for c in term)
    return has_digit or has_punct


def compute_salience_score(idf: float, df: int) -> float:
    """Score_sal(t) = IDF(t) * ln(1 + DF(t))"""
    return idf * math.log(1.0 + max(df, 0))


def compute_specificity_score(idf: float, cf: int) -> float:
    """Score_spec(t) = IDF(t) * ln(1 + CF(t))"""
    return idf * math.log(1.0 + max(cf, 0))


def compute_hybrid_score(idf: float, cf: int, df: int, num_docs: int) -> float:
    """Score_hyb(t) = IDF(t) * ln(1 + CF(t)) * (1 - DF(t) / N)"""
    return idf * math.log(1.0 + max(cf, 0)) * max(1.0 - (df / max(num_docs, 1)), 0.0)


def generate_salience_sequence(
    eligible_terms: List[str],
    df_map: Dict[str, int],
    idf_map: Dict[str, float],
    max_cap: int = MAX_CAPACITY,
) -> List[str]:
    """Sorts terms by Score_sal descending with deterministic lexicographic tie-breaking."""
    scored = [
        (compute_salience_score(idf_map.get(t, 0.0), df_map.get(t, 0)), t)
        for t in eligible_terms
    ]
    # Sort by score descending, term ascending (lexicographic tie-breaker)
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [t for _, t in scored[:max_cap]]


def generate_specificity_sequence(
    eligible_terms: List[str],
    cf_map: Dict[str, int],
    idf_map: Dict[str, float],
    max_cap: int = MAX_CAPACITY,
) -> List[str]:
    """Sorts terms by Score_spec descending with deterministic lexicographic tie-breaking."""
    scored = [
        (compute_specificity_score(idf_map.get(t, 0.0), cf_map.get(t, 0)), t)
        for t in eligible_terms
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [t for _, t in scored[:max_cap]]


def generate_hybrid_sequence(
    eligible_terms: List[str],
    cf_map: Dict[str, int],
    df_map: Dict[str, int],
    idf_map: Dict[str, float],
    num_docs: int,
    max_cap: int = MAX_CAPACITY,
) -> List[str]:
    """Sorts terms by Score_hyb descending with deterministic lexicographic tie-breaking."""
    scored = [
        (compute_hybrid_score(idf_map.get(t, 0.0), cf_map.get(t, 0), df_map.get(t, 0), num_docs), t)
        for t in eligible_terms
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [t for _, t in scored[:max_cap]]


def generate_stratified_sequence(
    eligible_terms: List[str],
    df_map: Dict[str, int],
    idf_map: Dict[str, float],
    max_cap: int = MAX_CAPACITY,
) -> List[str]:
    """
    Constructs a deterministic sequence using disjoint strata:
    1. S4 (Index-visible compounds/alphanumerics): Quota 10%
    2. S1 (High Specificity, spec >= 0.75): Quota 30%
    3. S2 (Medium Specificity, 0.40 <= spec < 0.75): Quota 40%
    4. S3 (Bridge Concepts, spec < 0.40): Quota 20%
    Ranked within strata by Score_sal(t) with lexicographic tie-breaking.
    Deterministically interleaved in proportional round-robin.
    """
    if not eligible_terms:
        return []

    idf_max = max((idf_map.get(t, 0.0) for t in eligible_terms), default=1.0)
    if idf_max <= 0.0:
        idf_max = 1.0

    s4, s1, s2, s3 = [], [], [], []

    for t in eligible_terms:
        if is_index_compound_or_alphanumeric(t):
            s4.append(t)
        else:
            spec = idf_map.get(t, 0.0) / idf_max
            if spec >= 0.75:
                s1.append(t)
            elif spec >= 0.40:
                s2.append(t)
            else:
                s3.append(t)

    # Rank within each stratum by Score_sal descending, term ascending
    def sort_stratum(stratum: List[str]) -> List[str]:
        return sorted(
            stratum,
            key=lambda t: (-compute_salience_score(idf_map.get(t, 0.0), df_map.get(t, 0)), t),
        )

    s1_sorted = sort_stratum(s1)
    s2_sorted = sort_stratum(s2)
    s3_sorted = sort_stratum(s3)
    s4_sorted = sort_stratum(s4)

    # Round-robin interleaved pattern based on integer weights:
    # S1: 3, S2: 4, S3: 2, S4: 1 (Total period = 10)
    cycle = [
        ("s1", s1_sorted), ("s2", s2_sorted), ("s1", s1_sorted), ("s2", s2_sorted),
        ("s3", s3_sorted), ("s1", s1_sorted), ("s2", s2_sorted), ("s3", s3_sorted),
        ("s2", s2_sorted), ("s4", s4_sorted),
    ]

    pointers = {"s1": 0, "s2": 0, "s3": 0, "s4": 0}
    lengths = {"s1": len(s1_sorted), "s2": len(s2_sorted), "s3": len(s3_sorted), "s4": len(s4_sorted)}
    
    result = []
    selected_set = set()

    # Step 1: Interleave while active items remain in cycle
    while len(result) < max_cap:
        progress = False
        for stratum_name, stratum_list in cycle:
            ptr = pointers[stratum_name]
            while ptr < lengths[stratum_name]:
                term = stratum_list[ptr]
                ptr += 1
                pointers[stratum_name] = ptr
                if term not in selected_set:
                    result.append(term)
                    selected_set.add(term)
                    progress = True
                    break
            if len(result) >= max_cap:
                break
        if not progress:
            break

    # Step 2: If quotas exhausted early, drain any remaining terms by Score_sal
    if len(result) < max_cap:
        remaining = [t for t in eligible_terms if t not in selected_set]
        remaining_sorted = sort_stratum(remaining)
        for t in remaining_sorted:
            result.append(t)
            selected_set.add(t)
            if len(result) >= max_cap:
                break

    return result[:max_cap]


def generate_celf_coverage_sequence(
    eligible_terms: List[str],
    doc_posting_supplier: Callable[[str], Set[int]],
    idf_map: Dict[str, float],
    df_map: Optional[Dict[str, int]] = None,
    max_cap: int = MAX_CAPACITY,
) -> List[str]:
    """
    CELF (Cost-Effective Lazy Forward) submodular greedy coverage selection.
    Exploits the submodularity of document set union: Delta_k+1(t) <= Delta_k(t).
    Maintains a max-heap of marginal gains. Evaluates posting lists only when
    the top candidate's cached upper bound exceeds the second best bound.
    Guarantees deterministic lexicographic tie-breaking.
    """
    if not eligible_terms or max_cap <= 0:
        return []

    # Priority queue stores (-gain_upper_bound, term_string, step_last_updated)
    pq = []
    for t in eligible_terms:
        idf_t = idf_map.get(t, 0.0)
        if idf_t <= 0.0:
            continue
        # Initial upper bound is idf * initial document count (df_map at step 0)
        if df_map is not None:
            initial_count = df_map.get(t, 0)
        else:
            initial_count = len(doc_posting_supplier(t))
        initial_gain = idf_t * initial_count
        heapq.heappush(pq, (-initial_gain, t, 0))

    selected_sequence = []
    covered_docs: Set[int] = set()
    step = 0

    while pq and len(selected_sequence) < max_cap:
        neg_bound, top_term, last_step = heapq.heappop(pq)
        bound = -neg_bound

        # If bound was computed in current step, it is provably optimal
        if last_step == step:
            selected_sequence.append(top_term)
            doc_ids = doc_posting_supplier(top_term)
            covered_docs.update(doc_ids)
            step += 1
            continue

        # Recompute exact marginal gain with current covered_docs
        doc_ids = doc_posting_supplier(top_term)
        new_docs_count = len(doc_ids - covered_docs)
        exact_gain = idf_map.get(top_term, 0.0) * new_docs_count

        # If pq is empty or exact_gain >= second best bound in heap
        if not pq or exact_gain >= -pq[0][0]:
            selected_sequence.append(top_term)
            covered_docs.update(doc_ids)
            step += 1
        else:
            # Re-insert with updated gain bound
            heapq.heappush(pq, (-exact_gain, top_term, step))

    return selected_sequence[:max_cap]


def generate_naive_greedy_coverage_sequence(
    eligible_terms: List[str],
    doc_posting_supplier: Callable[[str], Set[int]],
    idf_map: Dict[str, float],
    max_cap: int = MAX_CAPACITY,
) -> List[str]:
    """
    Reference naïve greedy coverage implementation used exclusively for
    validating CELF mathematical equivalence in unit tests.
    """
    remaining = set(eligible_terms)
    covered_docs: Set[int] = set()
    selected_sequence = []

    # Pre-fetch postings
    postings_cache = {t: doc_posting_supplier(t) for t in eligible_terms}

    while remaining and len(selected_sequence) < max_cap:
        best_term = None
        best_gain = -1.0

        # Sort remaining lexicographically for deterministic tie-breaking
        for t in sorted(remaining):
            idf_t = idf_map.get(t, 0.0)
            if idf_t <= 0.0:
                continue
            new_count = len(postings_cache[t] - covered_docs)
            gain = idf_t * new_count
            if gain > best_gain:
                best_gain = gain
                best_term = t

        if best_term is None or best_gain <= 0.0:
            # Drain remaining terms lexicographically
            for t in sorted(remaining):
                selected_sequence.append(t)
                if len(selected_sequence) >= max_cap:
                    break
            break

        selected_sequence.append(best_term)
        covered_docs.update(postings_cache[best_term])
        remaining.remove(best_term)

    return selected_sequence[:max_cap]


def get_effective_capacities(
    num_eligible: int,
    fixed_budgets: List[int] = FIXED_BUDGETS,
    percentage_cutoffs: List[float] = PERCENTAGE_CUTOFFS,
    max_cap: int = MAX_CAPACITY,
) -> Dict[str, int]:
    """
    Computes all effective capacity limits, deduplicating only when a percentage cutoff
    matches an existing fixed budget exactly in integer terms.
    """
    capacities: Dict[str, int] = {}

    for b in fixed_budgets:
        if b <= max_cap:
            capacities[f"{b // 1000}k" if b % 1000 == 0 else str(b)] = min(b, num_eligible)

    for p in percentage_cutoffs:
        p_cap = min(max_cap, int(math.floor(p * num_eligible)))
        p_label = f"{int(p * 100)}%"
        # Check exact deduplication with fixed budgets
        is_dup = any(val == p_cap for val in capacities.values())
        if not is_dup:
            capacities[p_label] = p_cap

    return capacities
