"""Low-cost ADE20K experiment with Qwen-VL.

- Validation: run K=1 and K=3 on one labeled ADE20K validation image.
- Testing: run K=3 qualitative inference on two unlabeled testing images.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from ugvu.configs.config import UGVUConfig
from ugvu.datasets.dataset import ADE20K_CLASSES
from ugvu.generators.sampler import build_sampler
from ugvu.decoders.decoder import build_decoder
from ugvu.pipeline import UGVUPipeline
from ugvu.uncertainty.uncertainty_map import build_uncertainty_map
from ugvu.consensus.consensus_fusion import consensus_fusion
from ugvu.visualization.visualize import save_uncertainty_map
from ugvu.diagnostics import (
    aggregate_failure_tags,
    attach_k_sweep_summary,
    prediction_diagnostics,
    semantic_trivial_baselines,
)

ADE_ROOT = r"D:\BaiduNetdiskDownload\ADE20K"
TEST_DIR = Path(r"D:\BaiduNetdiskDownload\ADE20K\images\testing")
OUT_ROOT = Path("outputs/ade20k_api")


def qwen_vl_model_config() -> dict:
    return {
        "qwen-vl": {
            "api_endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "${DASHSCOPE_API_KEY}",
            "model_version": "qwen-vl-max",
            "api_mode": "chat",
            "timeout_sec": 120,
            "max_retries": 1,
            "image_size": [1024, 1024],
            "temperature": 0.2,
        }
    }


def base_config(k: int, output_dir: str) -> dict:
    return {
        "experiment_name": f"ugvu_ade20k_qwenvl_k{k}",
        "seed": 42,
        "output_dir": output_dir,
        "models": qwen_vl_model_config(),
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
            "cross_model_models": ["qwen-vl"],
            "iou_threshold": 0.5,
        },
        "calibration": {"num_bins": 15, "strategy": "uniform", "metrics": ["ece", "mce", "spearman"]},
        "robustness": {"num_prompt_variants": 5, "generation_repeats": 2, "models_to_compare": ["qwen-vl"], "k_values": [1, 3]},
        "task": {
            "task": "semantic_segmentation",
            "dataset": "ade20k",
            "data_root": ADE_ROOT,
            "num_classes": 150,
            "ignore_index": 255,
            "batch_size": 1,
            "num_workers": 0,
            "max_samples": 1,
        },
    }


def run_validation() -> list[dict]:
    results = []
    for k in (1, 3):
        out_dir = OUT_ROOT / f"validation_qwenvl_k{k}"
        cfg = base_config(k, str(out_dir).replace("\\", "/"))
        cfg_path = OUT_ROOT / f"validation_qwenvl_k{k}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        pipe = UGVUPipeline(str(cfg_path), task="semantic_segmentation", output_dir=str(out_dir))
        item = {"split": "validation", "model": "qwen-vl", "k": k, "ok": False, "error": "", "metrics": {}, "diagnostics": {}}
        try:
            metrics = pipe.run()
            calib = pipe.run_calibration()
            item["ok"] = True
            item["metrics"] = {
                "mIoU": metrics.get("mIoU"),
                "ECE": calib.get("ece"),
                "Spearman": calib.get("spearman_r"),
                "AUROC": calib.get("auroc"),
            }
            per_image = [
                prediction_diagnostics(pred, gt, unc, ignore_index=255, num_classes=150)
                for pred, gt, unc in zip(pipe.predictions, pipe.ground_truths, pipe.uncertainty_maps)
            ]
            item["diagnostics"] = {
                "per_image": per_image,
                "failure_tag_counts": aggregate_failure_tags(per_image),
                "trivial_baselines": semantic_trivial_baselines(pipe.ground_truths, num_classes=150, ignore_index=255),
            }
        except Exception as exc:
            item["error"] = str(exc)
        results.append(item)
    return results


def ade_prompt() -> str:
    class_spec = ", ".join(f"{i}={name}" for i, name in enumerate(ADE20K_CLASSES))
    return (
        "Perform ADE20K semantic segmentation. Assign every pixel exactly one class ID from this list: "
        f"{class_spec}. Return a dense class-index mask, not a binary mask. "
        "Use only integers 0 through 149. Preserve major scene regions and object boundaries."
    )


def save_test_figure(image: np.ndarray, pred: np.ndarray, unc: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(pred, cmap="tab20", vmin=0, vmax=149)
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    im = axes[2].imshow(unc, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Uncertainty")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_testing(max_images: int = 2, k: int = 3) -> list[dict]:
    cfg = UGVUConfig.from_yaml("configs/default.yaml")
    model_cfg = qwen_vl_model_config()["qwen-vl"]
    model_cfg["api_key"] = cfg.models.models["qwen-vl"].api_key
    sampler = build_sampler({"qwen-vl": model_cfg}, k_samples=k, parallel=False, max_concurrent=1, seed=42, cache_dir=str(OUT_ROOT / "testing_qwenvl_k3" / "cache"))
    decoder = build_decoder("semantic_segmentation", num_classes=150, ignore_index=255, colormap=None)
    prompt = ade_prompt()
    out_dir = OUT_ROOT / "testing_qwenvl_k3"
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "uncertainties").mkdir(parents=True, exist_ok=True)
    (out_dir / "visualizations").mkdir(parents=True, exist_ok=True)

    results = []
    for idx, img_path in enumerate(sorted(TEST_DIR.glob("*.jpg"))[:max_images]):
        item = {"split": "testing", "image": str(img_path), "ok": False, "error": "", "classes": []}
        try:
            image = Image.open(img_path).convert("RGB")
            collection = sampler.sample(image, prompts=[prompt] * k, models=["qwen-vl"], image_id=img_path.stem)
            mask_stack, decode_unc = decoder.decode_batch_with_uncertainty(collection.mask_stack)
            unc = build_uncertainty_map(mask_stack, task="semantic_segmentation", uncertainty_type="entropy", num_classes=150, ignore_index=255)
            unc = np.maximum(unc, decode_unc.mean(axis=0))
            per_unc = np.maximum(decode_unc, np.tile(unc[None, ...], (mask_stack.shape[0], 1, 1)))
            pred = consensus_fusion(mask_stack, per_unc, method="ugcf", num_classes=150, ignore_index=255, temperature=0.5)
            np.save(out_dir / "predictions" / f"{img_path.stem}.npy", pred)
            np.save(out_dir / "uncertainties" / f"{img_path.stem}.npy", unc)
            save_uncertainty_map(unc, str(out_dir / "uncertainties" / f"{img_path.stem}.png"))
            save_test_figure(np.array(image), pred, unc, out_dir / "visualizations" / f"{img_path.stem}.png")
            classes, counts = np.unique(pred[pred != 255], return_counts=True)
            top = sorted(zip(classes.tolist(), counts.tolist()), key=lambda x: -x[1])[:8]
            item["classes"] = [{"id": int(c), "name": ADE20K_CLASSES[int(c)], "pixels": int(n)} for c, n in top]
            item["ok"] = True
        except Exception as exc:
            item["error"] = str(exc)
        results.append(item)
    return results


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = {"validation": run_validation(), "testing": run_testing()}
    with (OUT_ROOT / "ade20k_api_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    report = {
        "dataset": "ade20k",
        "protocol": "open_class_api",
        "paper_framing": "Validation diagnostics quantify zero-shot dense grounding failure; testing remains qualitative because labels are unavailable.",
        "validation_k_sweep_summary": attach_k_sweep_summary(results["validation"]),
        "validation": results["validation"],
        "testing": results["testing"],
    }
    with (OUT_ROOT / "ade20k_api_diagnostic_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

