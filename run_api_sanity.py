"""Small real-API sanity check for UGVU generators.

This script intentionally uses the synthetic Cityscapes fixture so it can test
remote generator connectivity without requiring the full Cityscapes dataset.
API keys are read from environment variables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ugvu.pipeline import UGVUPipeline


RIGHT_CODE_MODELS = (
    "gpt-image-2-vip",
    "gpt-image-2",
    "nano-banana",
    "nano-banana-2",
    "nano-banana-pro",
)

def _right_code_config(model_version: str) -> dict:
    return {
        "class": "rightcode",
        "api_endpoint": "https://www.right.codes/draw/v1",
        "api_key": "${IMAGE_API_KEY}",
        "model_version": model_version,
        "api_mode": "chat_completions",
        "timeout_sec": 300,
        "max_retries": 1,
        "image_size": [1024, 1024],
        "size": "1K",
        "temperature": 0.2,
    }


def _base_config() -> dict:
    right_code_models = {name: _right_code_config(name) for name in RIGHT_CODE_MODELS}
    return {
        "experiment_name": "ugvu_api_sanity",
        "seed": 42,
        "output_dir": "outputs/api_sanity",
        "models": {
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
            "qwen": {
                "api_endpoint": "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation",
                "api_key": "${DASHSCOPE_API_KEY}",
                "model_version": "qwen-image-2.0-pro",
                "api_mode": "generation",
                "timeout_sec": 120,
                "max_retries": 1,
                "image_size": [1024, 1024],
                "temperature": 0.2,
            },
            "google": {
                "api_endpoint": "https://generativelanguage.googleapis.com/v1beta",
                "api_key": "${GOOG_API_KEY}",
                "model_version": "gemini-2.0-flash",
                "api_mode": "generate_content",
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
            **right_code_models,
        },
        "sampling": {
            "k_samples": 1,
            "parallel": False,
            "max_concurrent": 1,
            "seed": 42,
            "output_dir": "outputs/samples",
            "adaptive": False,
            "min_samples": 1,
            "uncertainty_threshold": 0.15,
            "check_interval": 1,
        },
        "consensus": {
            "method": "majority",
            "uncertainty_type": "entropy",
            "weight_temperature": 0.5,
            "cross_model_models": ["doubao", "qwen"],
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
            "models_to_compare": ["doubao", "qwen"],
            "k_values": [1],
        },
        "task": {
            "task": "semantic_segmentation",
            "dataset": "cityscapes",
            "data_root": "data/synthetic_cityscapes",
            "num_classes": 19,
            "ignore_index": 255,
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": 1,
        },
    }


def run_one(model_name: str) -> dict:
    cfg = _base_config()
    cfg["models"] = {model_name: cfg["models"][model_name]}
    cfg["consensus"]["cross_model_models"] = [model_name]
    cfg_path = Path(f"outputs/api_sanity_{model_name}.yaml")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    pipe = UGVUPipeline(
        str(cfg_path),
        task="semantic_segmentation",
        output_dir=f"outputs/api_sanity_{model_name}",
    )
    result = {
        "model": model_name,
        "ok": False,
        "error": "",
        "metrics": {},
    }
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


def _parse_models(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-API sanity checks on synthetic Cityscapes.")
    parser.add_argument(
        "--models",
        default=",".join(RIGHT_CODE_MODELS),
        help="Comma-separated model names to run.",
    )
    args = parser.parse_args()
    available = _base_config()["models"]
    models = _parse_models(args.models)
    bad = [model for model in models if model not in available]
    if bad:
        raise ValueError(f"Unknown models: {bad}. Available: {list(available)}")
    results = [run_one(model) for model in models]
    out = Path("outputs/api_sanity_results.json")
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


