"""Experiment 2: Prompt Robustness.

Tests stability across N=50 semantically-equivalent prompt variants.

Computes PRS = 1 - σ/μ and analyzes per-prompt metric distribution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

from ugvu.pipeline import UGVUPipeline
from ugvu.prompts.prompt_pool import get_prompt_pool, generate_prompt_variants
from ugvu.robustness.prompt_variance import prompt_robustness_score


def run_exp2(config_path: str = "configs/default.yaml", num_prompts: int = 50):
    """Run Exp2: Prompt Robustness evaluation.

    Generates N prompt variants, runs inference with each, and computes PRS.
    """
    pipeline = UGVUPipeline(config_path, output_dir="outputs/exp2_prompt_robustness")
    pipeline._init_components()

    prompt_pool = get_prompt_pool("semantic_segmentation")
    variants = prompt_pool.generate_variants(n=num_prompts, seed=42)

    model_name = list(pipeline.sampler.generators.keys())[0]
    gen = pipeline.sampler.generators[model_name]
    decoder = pipeline.decoder
    tcfg = pipeline.config.task

    from ugvu.metrics.metrics import segmentation_metrics

    metrics_per_prompt = []
    for p_idx, prompt in enumerate(variants):
        print(f"\nPrompt {p_idx+1}/{num_prompts}: {prompt[:80]}...")

        preds = []
        gts = []
        for idx in range(min(len(pipeline.dataset), 50)):  # limit for speed
            sample = pipeline.dataset[idx]
            img = sample["image"]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            result = gen.generate(img, prompt)
            mask = np.array(result)
            pred = decoder.decode(mask)
            preds.append(pred)
            gts.append(sample["label"])

        ious = []
        for pred, gt in zip(preds, gts):
            ious.append(segmentation_metrics(pred, gt, tcfg.num_classes, tcfg.ignore_index)["mIoU"])

        mean_iou = np.mean(ious)
        metrics_per_prompt.append(mean_iou)
        print(f"  Mean mIoU = {mean_iou:.4f}")

    metrics_arr = np.array(metrics_per_prompt)
    prs = prompt_robustness_score(metrics_arr)

    print(f"\n{'='*60}")
    print("Exp2 Results: Prompt Robustness")
    print(f"{'='*60}")
    print(f"  PRS = {prs:.4f}")
    print(f"  Mean mIoU = {metrics_arr.mean():.4f} ± {metrics_arr.std():.4f}")
    print(f"  Min mIoU  = {metrics_arr.min():.4f}")
    print(f"  Max mIoU  = {metrics_arr.max():.4f}")

    return {"prs": prs, "per_prompt_metrics": metrics_per_prompt, "prompts": variants}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/default.yaml")
    parser.add_argument("--num-prompts", "-p", type=int, default=50)
    args = parser.parse_args()
    run_exp2(args.config, args.num_prompts)
