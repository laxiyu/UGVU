"""Small real-API experiment for UGVU.

Runs a low-cost comparison on the synthetic Cityscapes fixture:
  - Doubao Vision Chat, K=1 and K=3
  - Qwen-VL Chat, K=1 and K=3

The script reads API keys from .env/environment through UGVUConfig.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ugvu.pipeline import UGVUPipeline


MODEL_CONFIGS = {
    "doubao": {
        "api_endpoint": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key": "${DOUBAO_API_KEY}",
        "model_version": "ep-m-20260616184522-wp9rq",
        "api_mode": "vision_chat",
        "timeout_sec": 90,
        "max_retries": 1,
        "image_size": [160, 96],
        "temperature": 0.2,
    },
    "qwen-vl": {
        "api_endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "${DASHSCOPE_API_KEY}",
        "model_version": "qwen-vl-max",
        "api_mode": "chat",
        "timeout_sec": 90,
        "max_retries": 1,
        "image_size": [160, 96],
        "temperature": 0.2,
    },
}


def make_config(model_name: str, k: int) -> dict:
    return {
        "experiment_name": f"ugvu_api_{model_name}_k{k}",
        "seed": 42,
        "output_dir": f"outputs/api_exp_{model_name}_k{k}",
        "models": {model_name: MODEL_CONFIGS[model_name]},
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
            "data_root": "data/synthetic_cityscapes",
            "num_classes": 19,
            "ignore_index": 255,
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": 3,
        },
    }


def run_case(model_name: str, k: int) -> dict:
    cfg = make_config(model_name, k)
    cfg_path = Path(f"outputs/api_exp_{model_name}_k{k}.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    pipe = UGVUPipeline(str(cfg_path), task="semantic_segmentation", output_dir=cfg["output_dir"])
    result = {"model": model_name, "k": k, "ok": False, "error": "", "metrics": {}}
    try:
        metrics = pipe.run()
        calib = pipe.run_calibration()
        result["ok"] = True
        result["metrics"] = {
            "mIoU": metrics.get("mIoU"),
            "ECE": calib.get("ece"),
            "Spearman": calib.get("spearman_r"),
            "AUROC": calib.get("auroc"),
        }
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main() -> None:
    cases = [("doubao", 1), ("doubao", 3), ("qwen-vl", 1), ("qwen-vl", 3)]
    results = [run_case(model, k) for model, k in cases]
    out = Path("outputs/api_experiment_results.json")
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
