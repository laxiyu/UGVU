"""Candidate-constrained ADE20K experiment.

This is a stronger and more realistic protocol than asking a VLM to output all
150 ADE20K classes directly. For validation we use oracle candidates from the
GT mask as an upper-bound study: the model predicts local candidate IDs, then
we map them back to ADE20K IDs for mIoU/ECE evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ugvu.configs.config import UGVUConfig
from ugvu.datasets.dataset import ADE20K_CLASSES, build_dataset
from ugvu.generators.sampler import build_sampler
from ugvu.decoders.decoder import build_decoder
from ugvu.uncertainty.uncertainty_map import build_uncertainty_map
from ugvu.consensus.consensus_fusion import consensus_fusion
from ugvu.metrics.metrics import segmentation_metrics
from ugvu.calibration.ece import compute_ece_from_uncertainty
from ugvu.calibration.correlation import correlation_report
from ugvu.visualization.visualize import save_uncertainty_map
from ugvu.diagnostics import (
    aggregate_failure_tags,
    attach_k_sweep_summary,
    prediction_diagnostics,
    semantic_trivial_baselines,
)

ADE_ROOT = r"D:\BaiduNetdiskDownload\ADE20K"
OUT_ROOT = Path("outputs/ade20k_candidate")

MODELS = {
    "qwen-vl": {
        "api_endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "${DASHSCOPE_API_KEY}",
        "model_version": "qwen-vl-max",
        "api_mode": "chat",
        "timeout_sec": 120,
        "max_retries": 1,
        "image_size": [1024, 1024],
        "temperature": 0.2,
    },
    "doubao": {
        "api_endpoint": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key": "${DOUBAO_API_KEY}",
        "model_version": "ep-m-20260616184522-wp9rq",
        "api_mode": "vision_chat",
        "timeout_sec": 120,
        "max_retries": 1,
        "image_size": [1024, 1024],
        "temperature": 0.2,
    },
}


def load_keys() -> dict:
    cfg = UGVUConfig.from_yaml("configs/default.yaml")
    return {
        "qwen-vl": cfg.models.models["qwen-vl"].api_key,
        "doubao": cfg.models.models["doubao"].api_key,
    }


def candidate_prompt(candidates: list[int]) -> str:
    local = []
    for local_id, ade_id in enumerate(candidates):
        local.append(f"{local_id}={ADE20K_CLASSES[ade_id]} (ADE id {ade_id})")
    class_spec = "; ".join(local)
    return (
        "Perform ADE20K semantic segmentation using ONLY the candidate classes below. "
        "Output local candidate IDs, not ADE ids. "
        f"Candidate local IDs: {class_spec}. "
        "Every grid value must be one of the listed local IDs. "
        "Return a dense class-index grid; do not use unknown, background, 255, gradients, or color names. "
        "Make large coherent regions and preserve major object boundaries."
    )



def decode_candidate_stack(generated_stack: np.ndarray, candidates: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Decode local IDs while also accepting ADE original IDs.

    VLMs often ignore the instruction to emit local candidate IDs and instead
    emit ADE IDs. Treat both as valid and map everything back to local IDs.
    """
    local_stack = []
    unc_stack = []
    ade_to_local = {int(ade_id): i for i, ade_id in enumerate(candidates)}
    for generated in generated_stack:
        if generated.ndim == 3:
            values = generated[..., 0].astype(np.float32)
        else:
            values = generated.astype(np.float32)
        nearest = np.rint(values).astype(np.int64)
        local = np.full(nearest.shape, 255, dtype=np.int64)
        for local_id, ade_id in enumerate(candidates):
            local[nearest == local_id] = local_id
            local[nearest == int(ade_id)] = local_id
        quant_error = np.clip(np.abs(values - np.rint(values)), 0.0, 1.0).astype(np.float32)
        quant_error[local == 255] = 1.0
        local_stack.append(local)
        unc_stack.append(quant_error)
    return np.stack(local_stack, axis=0), np.stack(unc_stack, axis=0)

def map_local_to_ade(local_pred: np.ndarray, candidates: list[int]) -> np.ndarray:
    out = np.full(local_pred.shape, 255, dtype=np.int64)
    for local_id, ade_id in enumerate(candidates):
        out[local_pred == local_id] = ade_id
    return out


def save_comparison(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, unc: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(pred, cmap="tab20", vmin=0, vmax=149); axes[1].set_title("Prediction"); axes[1].axis("off")
    axes[2].imshow(gt, cmap="tab20", vmin=0, vmax=149); axes[2].set_title("GT"); axes[2].axis("off")
    im = axes[3].imshow(unc, cmap="hot", vmin=0, vmax=1); axes[3].set_title("Uncertainty"); axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_case(model_name: str, k: int, sample: dict, candidates: list[int], api_key: str) -> dict:
    out_dir = OUT_ROOT / f"{model_name}_k{k}"
    model_cfg = MODELS[model_name].copy()
    model_cfg["api_key"] = api_key
    sampler = build_sampler(
        {model_name: model_cfg},
        k_samples=k,
        parallel=False,
        max_concurrent=1,
        seed=42,
        cache_dir=str(out_dir / "cache"),
    )
    decoder = build_decoder("semantic_segmentation", num_classes=len(candidates), ignore_index=255, colormap=None)

    image_arr = sample["image"]
    gt = sample["label"].astype(np.int64)
    image = Image.fromarray(image_arr)
    prompt = candidate_prompt(candidates)

    result = {"model": model_name, "k": k, "ok": False, "error": "", "metrics": {}, "candidates": [], "diagnostics": {}}
    result["candidates"] = [{"local_id": i, "ade_id": int(c), "name": ADE20K_CLASSES[int(c)]} for i, c in enumerate(candidates)]
    try:
        collection = sampler.sample(image, prompts=[prompt] * k, models=[model_name], image_id="ade_val_00000001_candidate")
        local_stack, decode_unc = decode_candidate_stack(collection.mask_stack, candidates)
        unc = build_uncertainty_map(local_stack, task="semantic_segmentation", uncertainty_type="entropy", num_classes=len(candidates), ignore_index=255)
        unc = np.maximum(unc, decode_unc.mean(axis=0))
        per_unc = np.maximum(decode_unc, np.tile(unc[None, ...], (local_stack.shape[0], 1, 1)))
        local_pred = consensus_fusion(local_stack, per_unc, method="majority" if k == 1 else "ugcf", num_classes=len(candidates), ignore_index=255, temperature=0.5)
        pred = map_local_to_ade(local_pred, candidates)

        if pred.shape[:2] != gt.shape[:2]:
            pred = np.asarray(Image.fromarray(pred.astype(np.int32)).resize((gt.shape[1], gt.shape[0]), resample=Image.NEAREST), dtype=np.int64)
            unc = np.asarray(Image.fromarray(unc.astype(np.float32)).resize((gt.shape[1], gt.shape[0]), resample=Image.BILINEAR), dtype=np.float32)

        metrics = segmentation_metrics(pred, gt, num_classes=150, ignore_index=255)
        error_map = (pred != gt) & (gt != 255)
        ece = compute_ece_from_uncertainty(unc, error_map, num_bins=15, strategy="uniform")
        corr = correlation_report(unc, error_map)

        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "predictions").mkdir(exist_ok=True)
        (out_dir / "uncertainties").mkdir(exist_ok=True)
        np.save(out_dir / "predictions" / "pred_0000.npy", pred)
        np.save(out_dir / "uncertainties" / "unc_0000.npy", unc)
        save_uncertainty_map(unc, str(out_dir / "uncertainties" / "unc_0000.png"))
        save_comparison(image_arr, pred, gt, unc, out_dir / "comparison_0000.png")

        vals, counts = np.unique(pred[pred != 255], return_counts=True)
        result["pred_classes"] = [{"ade_id": int(v), "name": ADE20K_CLASSES[int(v)], "pixels": int(n)} for v, n in zip(vals, counts)]
        result["metrics"] = {
            "mIoU": metrics["mIoU"],
            "pixel_accuracy": metrics["pixel_accuracy"],
            "ECE": float(ece["ece"]),
            "Spearman": float(corr["spearman_r"]),
            "AUROC": float(corr["auroc"]),
        }
        diag = prediction_diagnostics(pred, gt, unc, ignore_index=255, num_classes=150)
        result["diagnostics"] = {
            "per_image": [diag],
            "per_image_calibration": [{
                "image": 0,
                "ECE": float(ece["ece"]),
                "MCE": float(ece["mce"]),
                "Spearman": float(corr["spearman_r"]),
                "AUROC": float(corr["auroc"]),
            }],
            "failure_tag_counts": aggregate_failure_tags([diag]),
            "trivial_baselines": semantic_trivial_baselines([gt], num_classes=150, ignore_index=255),
        }
        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset("ade20k", root=ADE_ROOT, split="val", max_samples=1)
    sample = dataset[0]
    candidates = sorted(int(v) for v in np.unique(sample["label"]) if int(v) != 255)
    keys = load_keys()
    results = []
    for model_name in ("qwen-vl", "doubao"):
        for k in (1, 3):
            results.append(run_case(model_name, k, sample, candidates, keys[model_name]))
    with (OUT_ROOT / "candidate_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    report = {
        "dataset": "ade20k",
        "protocol": "oracle_candidates",
        "max_samples": 1,
        "paper_framing": "Candidate-constrained diagnostic probing of zero-shot dense spatial reliability.",
        "k_sweep_summary": attach_k_sweep_summary(results),
        "results": results,
    }
    with (OUT_ROOT / "candidate_diagnostic_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


