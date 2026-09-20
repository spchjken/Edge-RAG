"""
src/pipeline_v2/selection/gate1_metrics.py

Formalized Evaluation Metrics and Data Transitions for Gate 1 Selection Under Uncertainty.
Directly implements Section 2 of docs/phase2_selection_under_uncertainty.md and the Approved Master Plan.
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

TAU: float = 1e-5
EPSILON: float = 0.001
DEFAULT_DELTA: float = 0.005
DEFAULT_RHO: float = 0.90
TRACKED_CUTOFFS: List[int] = [100, 200, 500, 1000]


def compute_safe_ranking_gain(
    actions: List[Dict[str, Any]],
    tau: float = TAU,
) -> float:
    r"""
    Computes safe individual ranking gain for term t:
    g^{rank}_{q,t} = max(0, max_{\mu : \Delta R@1000 >= -tau} \Delta nDCG@10).
    Returns 0.0 if no tested weight satisfies the safety constraint.
    """
    safe_deltas = [
        float(a.get("delta_ndcg10", 0.0))
        for a in actions
        if float(a.get("delta_r1000", 0.0)) >= -tau
    ]
    if not safe_deltas:
        return 0.0
    return max(0.0, float(np.max(safe_deltas)))


def compute_safe_recall_gain(
    actions: List[Dict[str, Any]],
    cutoff: int = 1000,
    epsilon: float = EPSILON,
) -> int:
    r"""
    Computes safe net recovered document count for term t at cutoff K:
    max(0, max_{\mu : \Delta nDCG@10 >= -epsilon} NetRelDocs@K).
    Returns 0 if no tested weight satisfies ranking safety.
    """
    net_col = f"net_rel_docs_k{cutoff}"
    safe_net = [
        int(a.get(net_col, 0))
        for a in actions
        if float(a.get("delta_ndcg10", 0.0)) >= -epsilon
    ]
    if not safe_net:
        return 0
    return max(0, int(np.max(safe_net)))


def get_ranking_helpful_terms(
    term_actions_map: Dict[str, List[Dict[str, Any]]],
    delta: float = DEFAULT_DELTA,
    tau: float = TAU,
) -> Set[str]:
    r"""
    Returns the set of materially ranking-helpful terms:
    H_q^{rank}(\delta) = {t \in R_q : g^{rank}_{q,t} >= \delta}.
    """
    helpful = set()
    for term, actions in term_actions_map.items():
        g = compute_safe_ranking_gain(actions, tau=tau)
        if g >= delta:
            helpful.add(term)
    return helpful


def get_recall_helpful_terms(
    term_actions_map: Dict[str, List[Dict[str, Any]]],
    cutoff: int = 1000,
    epsilon: float = EPSILON,
) -> Set[str]:
    r"""
    Returns the set of recall-helpful terms at cutoff K:
    H_q^{rec, K} = {t \in R_q : \exists \mu, NetRelDocs@K >= 1 and \Delta nDCG@10 >= -epsilon}.
    """
    helpful = set()
    for term, actions in term_actions_map.items():
        r = compute_safe_recall_gain(actions, cutoff=cutoff, epsilon=epsilon)
        if r >= 1:
            helpful.add(term)
    return helpful


def get_near_best_terms(
    term_actions_map: Dict[str, List[Dict[str, Any]]],
    ceiling_g: float,
    rho: float = DEFAULT_RHO,
    delta: float = DEFAULT_DELTA,
    tau: float = TAU,
) -> Set[str]:
    r"""
    Returns near-best terms:
    H_q^{near}(\rho, \delta) = {t \in R_q : g^{rank}_{q,t} >= max(\delta, \rho * ceiling_g)}.
    If ceiling_g < delta, returns empty set.
    """
    if ceiling_g < delta:
        return set()
    threshold = max(delta, rho * ceiling_g)
    near_best = set()
    for term, actions in term_actions_map.items():
        g = compute_safe_ranking_gain(actions, tau=tau)
        if g >= threshold:
            near_best.add(term)
    return near_best


def compute_reference_bor(
    proposed_terms: List[str],
    term_actions_map: Dict[str, List[Dict[str, Any]]],
    ceiling_g: float,
    tau: float = TAU,
) -> float:
    r"""
    Computes Reference Best-Opportunity Retention (ReferenceBOR@L):
    max_{t \in C_{1,L}} g^{rank}_{q,t} / ceiling_g  if ceiling_g > tau else NaN.
    """
    if ceiling_g <= tau:
        return np.nan
    if not proposed_terms:
        return 0.0
    best_prop = max(
        [compute_safe_ranking_gain(term_actions_map.get(t, []), tau=tau) for t in proposed_terms]
        + [0.0]
    )
    return float(best_prop / ceiling_g)


def compute_near_best_hit(
    proposed_terms: List[str],
    near_best_set: Set[str],
    ceiling_g: float,
    delta: float = DEFAULT_DELTA,
) -> float:
    r"""
    Returns NearBestHit@L:
    1 if C_{1,L} \cap H_q^{near} != \emptyset else 0, conditional on ceiling_g >= delta (else NaN).
    """
    if ceiling_g < delta:
        return np.nan
    hit = 1.0 if any(t in near_best_set for t in proposed_terms) else 0.0
    return hit


def compute_term_recall(
    proposed_terms: List[str],
    helpful_set: Set[str],
) -> float:
    r"""
    Computes TermRecall@L:
    |C_{1,L} \cap H_q| / |H_q| if |H_q| > 0 else NaN.
    """
    if not helpful_set:
        return np.nan
    matched = sum(1 for t in proposed_terms if t in helpful_set)
    return float(matched / len(helpful_set))


def compute_term_precision(
    proposed_terms: List[str],
    helpful_set: Set[str],
) -> float:
    r"""
    Computes TermPrecision@L:
    |C_{1,L} \cap H_q| / |C_{1,L}| if |C_{1,L}| > 0 else NaN.
    """
    if not proposed_terms:
        return np.nan
    matched = sum(1 for t in proposed_terms if t in helpful_set)
    return float(matched / len(proposed_terms))


def compute_recall_hit(
    proposed_terms: List[str],
    recall_helpful_set: Set[str],
    addressable_r_star: int,
) -> float:
    r"""
    Computes RecallHit@L,K:
    1 if C_{1,L} \cap H^{rec}_{q,K} != \emptyset else 0, conditional on r*_{q,K} >= 1 (else NaN).
    """
    if addressable_r_star < 1:
        return np.nan
    hit = 1.0 if any(t in recall_helpful_set for t in proposed_terms) else 0.0
    return hit


def compute_doc_opportunity_recall(
    proposed_terms: List[str],
    term_safe_doc_opp_map: Dict[str, Set[str]],
    universe_safe_doc_opp_union: Set[str],
) -> float:
    r"""
    Computes DocOpportunityRecall@L,K:
    |\bigcup_{t \in C_{1,L}} A^{safe}_{q,t,K}| / |\bigcup_{t \in R_q} A^{safe}_{q,t,K}|.
    Returns NaN if the denominator union is empty.
    """
    if not universe_safe_doc_opp_union:
        return np.nan
    prop_doc_union = set()
    for t in proposed_terms:
        prop_doc_union.update(term_safe_doc_opp_map.get(t, set()))
    return float(len(prop_doc_union) / len(universe_safe_doc_opp_union))


def compute_waste_metrics(
    proposed_terms: List[str],
    ceiling_g: float,
    delta: float = DEFAULT_DELTA,
    tau: float = TAU,
    l_max: int = 200,
) -> Dict[str, Any]:
    """
    Computes unaddressable candidate waste metrics:
    - waste_count: |C_1(q)|
    - is_unaddressable_material: bool (ceiling_g < delta)
    - is_unaddressable_strict: bool (ceiling_g <= tau)
    - normalized_waste: |C_1(q)| / l_max
    """
    count = len(proposed_terms)
    is_mat = ceiling_g < delta
    is_strict = ceiling_g <= tau
    return {
        "waste_count": count,
        "is_unaddressable_material": is_mat,
        "is_unaddressable_strict": is_strict,
        "normalized_waste": float(count / max(l_max, 1)),
    }


def classify_document_transition(
    baseline_rank: Optional[int],
    expanded_rank: Optional[int],
    cutoffs: List[int] = TRACKED_CUTOFFS,
) -> Optional[Dict[str, Any]]:
    """
    Classifies transition of a judged relevant document.
    Logs record if the document appears in baseline top-1000 or expanded top-1000
    AND its bounded rank or cutoff membership changed.
    
    Returns None if document is outside top-1000 in both runs or underwent zero change.
    """
    b_in_1000 = baseline_rank is not None and 1 <= baseline_rank <= 1000
    e_in_1000 = expanded_rank is not None and 1 <= expanded_rank <= 1000

    if not b_in_1000 and not e_in_1000:
        return None

    # Check rank delta
    rank_delta = None
    rank_changed = False
    if b_in_1000 and e_in_1000:
        rank_delta = int(baseline_rank - expanded_rank)
        rank_changed = rank_delta != 0
        trans_type = "moved_within_top1000"
    elif not b_in_1000 and e_in_1000:
        rank_changed = True
        trans_type = "entered_top1000"
    else:  # b_in_1000 and not e_in_1000
        rank_changed = True
        trans_type = "left_top1000"

    # Track cutoffs
    cutoff_flags = {}
    cutoff_directions = {}
    any_cutoff_crossed = False

    for k in cutoffs:
        b_in_k = baseline_rank is not None and 1 <= baseline_rank <= k
        e_in_k = expanded_rank is not None and 1 <= expanded_rank <= k
        crossed = b_in_k != e_in_k
        cutoff_flags[f"crossed_k{k}"] = crossed
        if crossed:
            any_cutoff_crossed = True
            direction = "entered" if e_in_k else "left"
        else:
            direction = "none"
        cutoff_directions[f"direction_k{k}"] = direction

    if not rank_changed and not any_cutoff_crossed:
        return None

    res = {
        "baseline_rank": baseline_rank if b_in_1000 else None,
        "expanded_rank": expanded_rank if e_in_1000 else None,
        "rank_delta": rank_delta,
        "transition_type": trans_type,
    }
    res.update(cutoff_flags)
    res.update(cutoff_directions)
    return res
