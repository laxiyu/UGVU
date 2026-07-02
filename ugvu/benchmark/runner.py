"""GVU-Robust Benchmark runner.

Orchestrates the full robustness evaluation suite:
    1. Prompt Robustness (PRS)
    2. Generation Robustness (GRS)
    3. Model Robustness (MRS)

Produces a comprehensive benchmark report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ..robustness.prompt_variance import evaluate_prompt_robustness, prompt_robustness_score
from ..robustness.generation_variance import evaluate_generation_robustness, generation_robustness_score
from ..robustness.model_variance import evaluate_model_robustness, model_robustness_score
from ..robustness.robustness_score import aggregate_robustness_report, robustness_score_str
from ..metrics.metrics import evaluate_dataset

logger = logging.getLogger(__name__)


class GVURobustBenchmark:
    """GVU-Robust Benchmark runner.

    Evaluates the robustness of a UGVU pipeline across three axes:
    prompt, generation, and model.

    Attributes:
        config: RobustnessConfig with benchmark parameters.
        output_dir: Directory for benchmark results.
    """

    def __init__(self, config, output_dir: str = "outputs/benchmark"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run(
        self,
        images: List[np.ndarray],
        ground_truths: List[np.ndarray],
        prompts: List[str],
        inference_fn: Callable,
        eval_fn: Callable,
        model_inference_fns: Optional[Dict[str, Callable]] = None,
    ) -> Dict:
        """Run the full GVU-Robust benchmark.

        Args:
            images: List of N input images.
            ground_truths: List of N ground truths.
            prompts: List of P prompt variants.
            inference_fn: function(image, prompt, seed=None) → prediction.
            eval_fn: function(prediction, ground_truth) → float metric.
            model_inference_fns: Optional dict of model_name → inference_fn.

        Returns:
            Full benchmark report dict.
        """
        print("\n" + "=" * 60)
        print("GVU-Robust Benchmark")
        print("=" * 60)

        # --- Prompt Robustness ---
        print("\n[1/3] Evaluating Prompt Robustness (PRS)...")
        prs_result = evaluate_prompt_robustness(
            images=images,
            prompts=prompts,
            inference_fn=lambda img, p: inference_fn(img, p, None),
            eval_fn=eval_fn,
            ground_truths=ground_truths,
            verbose=True,
        )
        self.results["prs"] = prs_result
        print(f"  PRS = {prs_result['prs_aggregate']:.4f}")

        # --- Generation Robustness ---
        print("\n[2/3] Evaluating Generation Robustness (GRS)...")
        k_values = self.config.k_values if hasattr(self.config, 'k_values') else [5, 10, 20]
        grs_by_k = {}
        for k in k_values:
            print(f"  K = {k}...")
            grs_result = evaluate_generation_robustness(
                images=images,
                prompt=prompts[0],
                inference_fn=lambda img, p, s: inference_fn(img, p, s),
                eval_fn=eval_fn,
                ground_truths=ground_truths,
                k_repeats=k,
                verbose=False,
            )
            grs_by_k[k] = grs_result
            print(f"    GRS(K={k}) = {grs_result['grs_aggregate']:.4f}")

        self.results["grs_by_k"] = grs_by_k
        self.results["grs"] = grs_by_k.get(20, list(grs_by_k.values())[-1])

        # --- Model Robustness ---
        print("\n[3/3] Evaluating Model Robustness (MRS)...")
        if model_inference_fns and len(model_inference_fns) > 1:
            mrs_result = evaluate_model_robustness(
                images=images,
                prompt=prompts[0],
                model_inference_fns=model_inference_fns,
                eval_fn=eval_fn,
                ground_truths=ground_truths,
                verbose=True,
            )
            self.results["mrs"] = mrs_result
            print(f"  MRS = {mrs_result['mrs_aggregate']:.4f}")
        else:
            self.results["mrs"] = None
            print("  MRS skipped (need ≥ 2 models).")

        # --- Aggregate ---
        report = aggregate_robustness_report(
            prs_result=prs_result,
            grs_result=self.results["grs"],
            mrs_result=self.results["mrs"],
        )
        self.results["summary"] = report

        print("\n" + robustness_score_str(report))

        # Save
        self._save_results()

        return self.results

    def _save_results(self):
        """Save benchmark results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        def _sanitize(obj):
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            return obj

        sanitized = _sanitize({
            "prs": {k: v for k, v in self.results.get("prs", {}).items() if k != "metric_matrix"},
            "grs_summary": {
                str(k): {
                    kk: vv for kk, vv in v.items() if kk != "metric_matrix"
                }
                for k, v in self.results.get("grs_by_k", {}).items()
            },
            "mrs": self.results.get("mrs"),
            "summary": self.results.get("summary"),
        })
        path = self.output_dir / "gvu_robust_results.json"
        with open(path, "w") as f:
            json.dump(sanitized, f, indent=2)
        print(f"\nResults saved to {path}")
