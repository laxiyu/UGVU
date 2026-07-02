"""Aggregate robustness scores — GVU-Robust benchmark summary.

Combines PRS, GRS, and MRS into a single report.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def aggregate_robustness_report(
    prs_result: Optional[Dict] = None,
    grs_result: Optional[Dict] = None,
    mrs_result: Optional[Dict] = None,
) -> Dict:
    """Combine PRS, GRS, MRS into a single robustness report.

    Args:
        prs_result: Output from prompt_robustness_per_image or evaluate_prompt_robustness.
        grs_result: Output from generation_robustness_per_image or evaluate_generation_robustness.
        mrs_result: Output from model_robustness_per_image or evaluate_model_robustness.

    Returns:
        Dict with:
            - prs: Prompt Robustness Score aggregate
            - grs: Generation Robustness Score aggregate
            - mrs: Model Robustness Score aggregate
            - gvu_robust: Overall GVU-Robust score (mean of available)
    """
    report = {}

    scores = []

    if prs_result:
        prs = prs_result.get("prs_aggregate", prs_result.get("prs_mean", None))
        report["prs"] = float(prs) if prs is not None else None
        if prs is not None:
            scores.append(float(prs))

    if grs_result:
        grs = grs_result.get("grs_aggregate", grs_result.get("grs_mean", None))
        report["grs"] = float(grs) if grs is not None else None
        if grs is not None:
            scores.append(float(grs))

    if mrs_result:
        mrs = mrs_result.get("mrs_aggregate", mrs_result.get("mrs_mean", None))
        report["mrs"] = float(mrs) if mrs is not None else None
        if mrs is not None:
            scores.append(float(mrs))

    report["gvu_robust"] = float(np.mean(scores)) if scores else None

    # Interpretation
    gvu = report["gvu_robust"]
    if gvu is not None:
        if gvu > 0.9:
            report["interpretation"] = "Excellent robustness — highly reliable across all axes."
        elif gvu > 0.7:
            report["interpretation"] = "Good robustness — reasonably stable, some variance."
        elif gvu > 0.5:
            report["interpretation"] = "Moderate robustness — notable sensitivity to perturbations."
        else:
            report["interpretation"] = "Poor robustness — predictions are unstable. Consider UGCF/CMCF."

    return report


def robustness_score_str(report: Dict) -> str:
    """Human-readable string summary of robustness."""
    lines = [
        "=" * 55,
        "GVU-Robust Benchmark Results",
        "=" * 55,
    ]
    if report.get("prs") is not None:
        lines.append(f"  PRS (Prompt Robustness):      {report['prs']:.4f}")
    if report.get("grs") is not None:
        lines.append(f"  GRS (Generation Robustness):   {report['grs']:.4f}")
    if report.get("mrs") is not None:
        lines.append(f"  MRS (Model Robustness):        {report['mrs']:.4f}")
    lines.append(f"  ──────────────────────────────────")
    if report.get("gvu_robust") is not None:
        lines.append(f"  GVU-Robust (overall):          {report['gvu_robust']:.4f}")
    if report.get("interpretation"):
        lines.append(f"\n  {report['interpretation']}")
    lines.append("=" * 55)
    return "\n".join(lines)
