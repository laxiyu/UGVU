"""Low-cost Cityscapes API experiment for UGVU.

Runs Qwen-VL and Doubao on a few Cityscapes val images with K=1/K=3.
Reads API keys from .env through UGVUConfig.

The result file keeps the original list format. A second diagnostic report adds
trivial baselines, K-sweep relative changes, and per-image failure tags for the
paper's reliability/probing narrative.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from ugvu.diagnostics import (
    aggregate_failure_tags,
    attach_k_sweep_summary,
    prediction_diagnostics,
    semantic_trivial_baselines,
)
from ugvu.pipeline import UGVUPipeline

CITYSCAPES_ROOT = r"D:\BaiduNetdiskDownload\cityscapes"
OUT_ROOT = Path("outputs/cityscapes_api")
MAX_SAMPLES = 3

MODELS = {
    "qwen-vl": {
        "api_endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "${DASHSCOPE_API_KEY}",
        "model_version": "qwen-vl-max",
        "api_mode": "chat",
        "timeout_sec": 120,
        "max_retries": 1,
        "image_size": [1024, 512],
        "temperature": 0.2,
    },
    "doubao": {
        "api_endpoint": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key": "${DOUBAO_API_KEY}",
        "model_version": "ep-m-20260616184522-wp9rq",
        "api_mode": "vision_chat",
        "timeout_sec": 120,
        "max_retries": 1,
        "image_size": [1024, 512],
        "temperature": 0.2,
    },
}


def make_config(model_name: str, k: int, output_dir: str) -> dict:
    return {
        "experiment_name": f"ugvu_cityscapes_{model_name}_k{k}",
        "seed": 42,
        "output_dir": output_dir,
        "models": {model_name: MODELS[model_name]},
        "sampling": {
            "k_samples": k,
            "parallel": False,
            "max_concurrent": 1,
            "seed": 42,
            "output_dir": "outputs/samples",
            "adaptive": False,
            "min_samples": min(3, k),
            "uncertainty_threshold": 0.15,
            "check_interval": 1,
        },
        "consensus": {
            "method": "majority" if k == 1 else "ugcf",
            "uncertainty_type": "entropy",
            "weight_temperature": 0.5,
            "cross_model_models": [model_name],
            "iou_threshold": 0.5,
        },
        "calibration": {
            "num_bins": 15,
            "strategy": "uniform",
            "metrics": ["ece", "mce", "spearman"],
        },
        "robustness": {
            "num_prompt_variants": 5,
            "generation_repeats": 2,
            "models_to_compare": [model_name],
            "k_values": [1, 3],
        },
        "task": {
            "task": "semantic_segmentation",
            "dataset": "cityscapes",
            "data_root": CITYSCAPES_ROOT,
            "num_classes": 19,
            "ignore_index": 255,
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": MAX_SAMPLES,
        },
    }


def run_case(model_name: str, k: int) -> dict:
    out_dir = OUT_ROOT / f"{model_name}_k{k}"
    cfg = make_config(model_name, k, str(out_dir).replace("\\", "/"))
    cfg_path = OUT_ROOT / f"{model_name}_k{k}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    pipe = UGVUPipeline(str(cfg_path), task="semantic_segmentation", output_dir=str(out_dir))
    item = {
        "dataset": "cityscapes",
        "model": model_name,
        "k": k,
        "max_samples": MAX_SAMPLES,
        "ok": False,
        "error": "",
        "metrics": {},
        "diagnostics": {},
    }
    try:
        metrics = pipe.run()
        if not pipe.predictions:
            raise RuntimeError("No successful predictions in this run; refusing to mix stale cached outputs into calibration.")
        calib = pipe.run_calibration()
        per_image = [
            prediction_diagnostics(pred, gt, unc, ignore_index=255, num_classes=19)
            for pred, gt, unc in zip(pipe.predictions, pipe.ground_truths, pipe.uncertainty_maps)
        ]
        item["ok"] = True
        item["metrics"] = {
            "mIoU": metrics.get("mIoU"),
            "ECE": calib.get("ece"),
            "Spearman": calib.get("spearman_r"),
            "AUROC": calib.get("auroc"),
        }
        item["diagnostics"] = {
            "per_image": per_image,
            "failure_tag_counts": aggregate_failure_tags(per_image),
            "trivial_baselines": semantic_trivial_baselines(pipe.ground_truths, num_classes=19, ignore_index=255),
        }
    except Exception as exc:
        item["error"] = str(exc)
    return item


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for model_name in ("qwen-vl", "doubao"):
        for k in (1, 3):
            results.append(run_case(model_name, k))

    with (OUT_ROOT / "cityscapes_api_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    report = {
        "dataset": "cityscapes",
        "protocol": "open_class_api",
        "max_samples": MAX_SAMPLES,
        "paper_framing": "Diagnostic probing of zero-shot dense spatial reliability, not SOTA segmentation.",
        "k_sweep_summary": attach_k_sweep_summary(results),
        "results": results,
    }
    with (OUT_ROOT / "cityscapes_api_diagnostic_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

