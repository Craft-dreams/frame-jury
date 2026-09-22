"""benchmarks.scoring — precision, recall, F1, and abstention scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.competitors import CompetitorResult


def score(
    results: dict[str, CompetitorResult],
    labels: dict[str, set[str]],
    defect: str,
) -> dict[str, float | None]:
    """Score competitor results against ground-truth labels for a given defect.

    Parameters
    ----------
    results:
        Mapping of case_id to CompetitorResult.
    labels:
        Mapping of case_id to set of ground-truth defect slugs/decisions.
    defect:
        The defect slug to score (e.g. 'duplicated_character').

    Returns
    -------
    dict[str, float | None]
        Dictionary with keys 'precision', 'recall', 'f1', 'support', 'abstain_rate',
        'excluded_legacy', 'tp', 'fp', 'fn'.
        Metrics whose denominator is zero return None (never 0.0).

    Rules (SPEC.md §7, §8):
    1. 'uncertain' cases are excluded from scoring entirely, never guessed.
    2. Legacy schema-1.0 'broken_anatomy' means "body or hands, unspecified".
       It is excluded from scoring both 'broken_body' and 'broken_hands', and
       reported in the excluded tally.
    3. An abstention is neither a positive nor a negative. Exclude the case from
       that defect's precision and recall, and report the abstention rate.
    """
    abstain_count = sum(1 for r in results.values() if defect in r.abstained)
    abstain_rate = (abstain_count / len(results)) if results else None

    # When zero labels are available, we cannot compute precision/recall/F1.
    if not labels:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "support": 0.0,
            "abstain_rate": abstain_rate,
            "excluded_legacy": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
        }

    tp = 0
    fp = 0
    fn = 0
    excluded_legacy = 0

    for case_id, res in results.items():
        if case_id not in labels:
            continue
        gt_defects = labels[case_id]

        # Rule 1: 'uncertain' cases excluded entirely
        if "uncertain" in gt_defects:
            continue

        # Rule 2: legacy 'broken_anatomy' excluded from broken_body and broken_hands
        if defect in ("broken_body", "broken_hands") and "broken_anatomy" in gt_defects:
            excluded_legacy += 1
            continue

        # Rule 3: abstention is neither positive nor negative
        if defect in res.abstained:
            continue

        is_gt_positive = defect in gt_defects
        is_pred_positive = defect in res.defects

        if is_pred_positive and is_gt_positive:
            tp += 1
        elif is_pred_positive and not is_gt_positive:
            fp += 1
        elif not is_pred_positive and is_gt_positive:
            fn += 1

    # Precision: TP / (TP + FP)
    pred_positives = tp + fp
    precision = (tp / pred_positives) if pred_positives > 0 else None

    # Recall: TP / (TP + FN)
    actual_positives = tp + fn
    recall = (tp / actual_positives) if actual_positives > 0 else None

    # F1: 2 * P * R / (P + R)
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    support = float(actual_positives)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "abstain_rate": abstain_rate,
        "excluded_legacy": float(excluded_legacy),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
