"""UGVU Pipeline — end-to-end orchestration.

This is the central coordinator that wires together all UGVU modules:
    Generation → Decoding → Uncertainty → Consensus → Calibration → Robustness

Usage:
    pipeline = UGVUPipeline("configs/default.yaml", task="semantic_segmentation")
    pipeline.run()
"""

from __future__ import annotations

import json
import time
import traceback
import concurrent.futures
from pathlib import Path
import logging
import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from .configs.config import UGVUConfig
from .datasets.dataset import build_dataset
from .prompts.prompt_pool import get_prompt_pool
from .generators.sampler import Sampler, build_sampler, SampleCollection
from .decoders.decoder import build_decoder, BaseDecoder
from .uncertainty.uncertainty_map import build_uncertainty_map, uncertainty_to_confidence
from .consensus.consensus_fusion import consensus_fusion, fuse_cross_model
from .calibration.ece import compute_ece_from_uncertainty
from .calibration.correlation import correlation_report
from .calibration.calibration_curve import plot_calibration_summary
from .metrics.metrics import evaluate_dataset, segmentation_metrics
from .robustness.robustness_score import aggregate_robustness_report, robustness_score_str
from .visualization.visualize import save_comparison_figure, save_uncertainty_map

logger = logging.getLogger(__name__)


class UGVUPipeline:
    """End-to-end UGVU pipeline.

    Orchestrates:
        1. Load dataset
        2. K-shot generation (multi-model)
        3. Decode generated images to structured predictions
        4. Compute Pixel-wise Consensus Uncertainty (PCU)
        5. Consensus fusion (UGCF or CMCF)
        6. Calibration analysis
        7. Robustness benchmark (GVU-Robust)
        8. Save results & visualizations
    """

    def __init__(
        self,
        config_path: str,
        task: str = "semantic_segmentation",
        output_dir: Optional[str] = None,
    ):
        self.config = UGVUConfig.from_yaml(config_path)
        self.task = task
        self.output_dir = Path(output_dir or self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set seed
        np.random.seed(self.config.seed)

        # Components (lazy init)
        self.sampler: Optional[Sampler] = None
        self.decoder: Optional[BaseDecoder] = None
        self.prompt_pool = get_prompt_pool(task)
        self.dataset = None

        # Results storage
        self.predictions: List[np.ndarray] = []
        self.uncertainty_maps: List[np.ndarray] = []
        self.ground_truths: List[np.ndarray] = []
        self.metrics_result: Dict = {}
        self.calibration_result: Dict = {}
        self.robustness_result: Dict = {}
        self.sample_counts: List[int] = []

    # ========================================================================
    # Initialization
    # ========================================================================

    def _init_components(self):
        """Lazy-initialize sampler, decoder, and dataset."""
        if self.sampler is None:
            model_cfgs = {}
            for name, mcfg in self.config.models.models.items():
                model_cfgs[name] = {
                    "api_endpoint": mcfg.api_endpoint,
                    "api_key": mcfg.api_key,
                    "model_version": mcfg.model_version,
                    "api_mode": mcfg.api_mode,
                    "timeout_sec": mcfg.timeout_sec,
                    "max_retries": mcfg.max_retries,
                    "image_size": mcfg.image_size,
                    "temperature": mcfg.temperature,
                    **mcfg.extra_params,
                }
            self.sampler = build_sampler(
                model_configs=model_cfgs,
                k_samples=self.config.sampling.k_samples,
                parallel=self.config.sampling.parallel,
                max_concurrent=self.config.sampling.max_concurrent,
                seed=self.config.sampling.seed,
                cache_dir=str(self.output_dir / "cache"),
            )

        if self.decoder is None:
            tcfg = self.config.task

            # 为 QwenGenerator (chat 模式) 设置 colormap
            from .datasets.dataset import DATASET_REGISTRY
            ds_cls = DATASET_REGISTRY.get(tcfg.dataset)
            cmap = ds_cls.colormap if hasattr(ds_cls, 'colormap') and not isinstance(ds_cls.colormap, property) else None
            class_names = ds_cls.class_names if hasattr(ds_cls, 'class_names') and not isinstance(ds_cls.class_names, property) else None
            if cmap is not None:
                for gen in self.sampler.generators.values():
                    if hasattr(gen, 'colormap'):
                        gen.colormap = cmap
            if self.task == "semantic_segmentation" and class_names:
                class_spec = ", ".join(f"{i}={name}" for i, name in enumerate(class_names[:tcfg.num_classes]))
                detailed_prompt = (
                    "Perform Cityscapes-style semantic segmentation. "
                    "Assign every pixel exactly one class ID from this list: "
                    f"{class_spec}. "
                    f"Do not use {tcfg.ignore_index} unless the pixel is truly impossible to assign. "
                    "Return a dense class-index mask, not a binary mask."
                )
                self.prompt_pool.base_templates = [
                    detailed_prompt,
                    detailed_prompt + " Preserve object boundaries and small objects such as cars, signs, and persons.",
                    detailed_prompt + " Use the road/sky/building/vegetation/vehicle IDs consistently.",
                ]
                self.prompt_pool.default_prompt = detailed_prompt
                self.prompt_pool.variants = []
                self.prompt_pool._variant_pointer = 0

            self.decoder = build_decoder(
                task=self.task,
                num_classes=tcfg.num_classes,
                ignore_index=tcfg.ignore_index,
                colormap=cmap,
            )

        if self.dataset is None:
            tcfg = self.config.task
            self.dataset = build_dataset(
                tcfg.dataset,
                root=tcfg.data_root,
                split="val",
                max_samples=tcfg.max_samples,
            )

    # ========================================================================
    # Single-sample processing (with error handling & incremental save)
    # ========================================================================

    def _process_single_sample(self, idx: int) -> Optional[tuple]:
        """处理单张图片的完整逻辑，包含异常捕获和即时落盘（防 OOM）。"""
        tcfg = self.config.task
        consensus_method = self.config.consensus.method
        use_cmcf = consensus_method == "cmcf"
        is_continuous = self.task == "depth_estimation"

        try:
            sample = self.dataset[idx]
            image = sample["image"]
            gt = sample["label"]
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # ---- 1. Generation ----
            prompts = [self.prompt_pool.next_variant() for _ in range(self.config.sampling.k_samples)]
            models = self.config.consensus.cross_model_models if use_cmcf else None

            # ---- 2. Decode ----
            if self.config.sampling.adaptive:
                collection, mask_stack, decode_unc_stack, uncertainty = self._adaptive_sample_decode(
                    image=image,
                    prompts=prompts,
                    models=models,
                    image_id=str(idx),
                    is_continuous=is_continuous,
                )
            else:
                collection = self.sampler.sample(image, prompts=prompts, models=models, image_id=str(idx))
                mask_stack, decode_unc_stack = self.decoder.decode_batch_with_uncertainty(collection.mask_stack)

            # ---- 3. Uncertainty ----
            unc_type = "entropy_continuous" if (
                self.config.consensus.uncertainty_type == "entropy" and is_continuous
            ) else self.config.consensus.uncertainty_type

            if not self.config.sampling.adaptive:
                uncertainty = build_uncertainty_map(
                    mask_stack, task=self.task, uncertainty_type=unc_type,
                    num_classes=tcfg.num_classes, ignore_index=tcfg.ignore_index,
                )
                if decode_unc_stack.shape == mask_stack.shape:
                    uncertainty = np.maximum(uncertainty, decode_unc_stack.mean(axis=0))

            # Per-sample uncertainty for UGCF weighting
            if decode_unc_stack.shape == mask_stack.shape:
                consensus_broadcast = np.tile(uncertainty[None, ...], (mask_stack.shape[0], 1, 1))
                per_sample_unc = np.maximum(decode_unc_stack, consensus_broadcast)
            elif mask_stack.ndim == 3 and not is_continuous:
                per_sample_unc = np.tile(uncertainty[None, ...], (mask_stack.shape[0], 1, 1))
            else:
                per_sample_unc = np.tile(uncertainty[None, ...], (mask_stack.shape[0], 1, 1))

            # ---- 4. Consensus ----
            if use_cmcf:
                model_masks_dict = {}
                model_uncs_dict = {}
                for k, model_name in enumerate(collection.model_names):
                    if model_name not in model_masks_dict:
                        model_masks_dict[model_name] = []
                        model_uncs_dict[model_name] = []
                    model_masks_dict[model_name].append(mask_stack[k])
                    model_uncs_dict[model_name].append(per_sample_unc[k])

                fused_per_model = {}
                fused_unc_per_model = {}
                for m_name in model_masks_dict:
                    m_masks = np.stack(model_masks_dict[m_name])
                    m_uncs = np.stack(model_uncs_dict[m_name])
                    fused_per_model[m_name] = consensus_fusion(
                        m_masks, m_uncs, method="ugcf",
                        num_classes=tcfg.num_classes, ignore_index=tcfg.ignore_index,
                        temperature=self.config.consensus.weight_temperature,
                        is_continuous=is_continuous,
                    )
                    fused_unc_per_model[m_name] = m_uncs.mean(axis=0)

                final_pred = fuse_cross_model(
                    fused_per_model, fused_unc_per_model,
                    num_classes=tcfg.num_classes, ignore_index=tcfg.ignore_index,
                    temperature=self.config.consensus.weight_temperature,
                    is_continuous=is_continuous,
                )
            else:
                final_pred = consensus_fusion(
                    mask_stack, per_sample_unc, method=consensus_method,
                    num_classes=tcfg.num_classes, ignore_index=tcfg.ignore_index,
                    temperature=self.config.consensus.weight_temperature,
                    is_continuous=is_continuous,
                )

            # Align API outputs to the ground-truth resolution for evaluation.
            if hasattr(gt, "shape") and final_pred.shape[:2] != gt.shape[:2]:
                target_hw = gt.shape[:2]
                if is_continuous:
                    final_pred = np.array(
                        Image.fromarray(final_pred.astype(np.float32)).resize(
                            (target_hw[1], target_hw[0]), resample=Image.BILINEAR
                        ),
                        dtype=np.float32,
                    )
                else:
                    final_pred = np.array(
                        Image.fromarray(final_pred.astype(np.int32)).resize(
                            (target_hw[1], target_hw[0]), resample=Image.NEAREST
                        ),
                        dtype=np.int64,
                    )
                uncertainty = np.array(
                    Image.fromarray(uncertainty.astype(np.float32)).resize(
                        (target_hw[1], target_hw[0]), resample=Image.BILINEAR
                    ),
                    dtype=np.float32,
                )
            # ---- 5. Incremental save ----
            pred_dir = self.output_dir / "predictions"
            unc_dir = self.output_dir / "uncertainties"
            vis_dir = self.output_dir / "visualizations"
            pred_dir.mkdir(parents=True, exist_ok=True)
            unc_dir.mkdir(parents=True, exist_ok=True)
            vis_dir.mkdir(parents=True, exist_ok=True)

            np.save(pred_dir / f"pred_{idx:04d}.npy", final_pred)
            np.save(unc_dir / f"unc_{idx:04d}.npy", uncertainty)
            save_uncertainty_map(uncertainty, str(unc_dir / f"unc_{idx:04d}.png"))

            # Save comparison figure
            try:
                img_arr = np.array(image) if not isinstance(image, np.ndarray) else image
                from .datasets.dataset import DATASET_REGISTRY
                ds_cls = DATASET_REGISTRY.get(tcfg.dataset)
                cmap = ds_cls.colormap if hasattr(ds_cls, 'colormap') and not isinstance(ds_cls.colormap, property) else None
                save_comparison_figure(
                    img_arr, final_pred, gt, uncertainty,
                    output_path=str(vis_dir / f"comparison_{idx:04d}.png"),
                    pred_colormap=cmap, gt_colormap=cmap,
                )
            except Exception:
                pass  # visualization failure is non-critical

            return final_pred, uncertainty, gt

        except Exception as e:
            logger.error(f"Failed to process image {idx}. Error: {e}\n{traceback.format_exc()}")
            return None

    def _adaptive_sample_decode(
        self,
        image: Image.Image,
        prompts: List[str],
        models: Optional[List[str]],
        image_id: str,
        is_continuous: bool,
    ) -> tuple[SampleCollection, np.ndarray, np.ndarray, np.ndarray]:
        """Sample until the consensus uncertainty is low enough or K is exhausted."""
        scfg = self.config.sampling
        tcfg = self.config.task
        unc_type = "entropy_continuous" if (
            self.config.consensus.uncertainty_type == "entropy" and is_continuous
        ) else self.config.consensus.uncertainty_type

        max_k = max(1, scfg.k_samples)
        min_k = min(max(1, scfg.min_samples), max_k)
        interval = max(1, scfg.check_interval)
        threshold = float(scfg.uncertainty_threshold)

        collection = SampleCollection(image_id=image_id, source_image=image)
        mask_stack = np.empty((0,), dtype=np.float32)
        decode_unc_stack = np.empty((0,), dtype=np.float32)
        uncertainty = np.empty((0,), dtype=np.float32)

        sampled = 0
        while sampled < max_k:
            target = min(max_k, min_k if sampled == 0 else sampled + interval)
            batch_k = target - sampled
            old_k = self.sampler.k_samples
            self.sampler.k_samples = batch_k
            try:
                batch = self.sampler.sample(
                    image,
                    prompts=prompts[sampled:target],
                    models=models,
                    image_id=image_id,
                    seed_offset=sampled,
                )
            finally:
                self.sampler.k_samples = old_k

            for mask, prompt, model, seed in zip(batch.masks, batch.prompts, batch.model_names, batch.seeds):
                collection.add_sample(mask, prompt=prompt, model=model, seed=seed)

            mask_stack, decode_unc_stack = self.decoder.decode_batch_with_uncertainty(collection.mask_stack)
            uncertainty = build_uncertainty_map(
                mask_stack,
                task=self.task,
                uncertainty_type=unc_type,
                num_classes=tcfg.num_classes,
                ignore_index=tcfg.ignore_index,
            )
            if decode_unc_stack.shape == mask_stack.shape:
                uncertainty = np.maximum(uncertainty, decode_unc_stack.mean(axis=0))

            sampled = collection.k
            if sampled >= min_k and float(np.mean(uncertainty)) <= threshold:
                break

        self.sample_counts.append(collection.k)
        return collection, mask_stack, decode_unc_stack, uncertainty

    # ========================================================================
    # Main pipeline
    # ========================================================================

    def run(self) -> Dict:
        """Run the full UGVU pipeline end-to-end."""
        self._init_components()
        tcfg = self.config.task

        logger.info(f"Running UGVU pipeline: {self.config.experiment_name}")
        logger.info(f"  Task: {self.task} | Dataset: {tcfg.dataset} | K={self.config.sampling.k_samples}")
        logger.info(f"  Consensus: {self.config.consensus.method} | Uncertainty: {self.config.consensus.uncertainty_type}")

        # Generate prompt variants
        self.prompt_pool.generate_variants(n=self.config.robustness.num_prompt_variants, seed=self.config.seed)

        predictions = []
        uncertainties = []
        ground_truths = []

        for idx in tqdm(range(len(self.dataset)), desc="Processing images"):
            result = self._process_single_sample(idx)
            if result is not None:
                final_pred, uncertainty, gt = result
                predictions.append(final_pred)
                uncertainties.append(uncertainty)
                ground_truths.append(gt)
            else:
                logger.warning(f"Skipped image {idx} due to processing error.")

        # ====================================================================
        # Evaluation
        # ====================================================================
        logger.info("Evaluating predictions...")
        self.predictions = predictions
        self.uncertainty_maps = uncertainties
        self.ground_truths = ground_truths
        self.metrics_result = evaluate_dataset(
            predictions, ground_truths,
            task=self.task,
            num_classes=tcfg.num_classes,
            ignore_index=tcfg.ignore_index,
        )
        if self.sample_counts:
            self.metrics_result["avg_samples_per_image"] = float(np.mean(self.sample_counts))
            self.metrics_result["max_samples_per_image"] = int(np.max(self.sample_counts))

        logger.info(f"Metrics: {json.dumps(self.metrics_result, indent=2)}")

        # ====================================================================
        # Save results
        # ====================================================================
        self._save_results()

        return self.metrics_result

    # ========================================================================
    # Evaluation only
    # ========================================================================

    def evaluate(self) -> Dict:
        """Run evaluation only (assumes predictions already loaded)."""
        tcfg = self.config.task
        if not self.predictions:
            # Try loading from disk
            self.load_results()
        if not self.predictions:
            raise RuntimeError("No predictions. Run pipeline first or load from disk.")
        self.metrics_result = evaluate_dataset(
            self.predictions, self.ground_truths,
            task=self.task,
            num_classes=tcfg.num_classes,
            ignore_index=tcfg.ignore_index,
        )
        return self.metrics_result

    # ========================================================================
    # Load results from disk
    # ========================================================================

    def load_results(self) -> bool:
        """Load predictions, uncertainties and ground truths from disk.

        Returns True if results were successfully loaded.
        """
        pred_dir = self.output_dir / "predictions"
        unc_dir = self.output_dir / "uncertainties"

        if not pred_dir.exists():
            logger.warning(f"No predictions directory found at {pred_dir}")
            return False

        pred_files = sorted(pred_dir.glob("pred_*.npy"))
        if not pred_files:
            logger.warning(f"No prediction files found in {pred_dir}")
            return False

        self.predictions = [np.load(f) for f in pred_files]
        logger.info(f"Loaded {len(self.predictions)} predictions from {pred_dir}")

        # Load uncertainties
        unc_files = sorted(unc_dir.glob("unc_*.npy")) if unc_dir.exists() else []
        if unc_files:
            self.uncertainty_maps = [np.load(f) for f in unc_files]

        # Load ground truths from dataset if available
        if not self.ground_truths and self.dataset is None:
            try:
                self._init_components()
            except Exception:
                pass

        if self.dataset is not None and not self.ground_truths:
            self.ground_truths = []
            for idx in range(min(len(self.predictions), len(self.dataset))):
                sample = self.dataset[idx]
                self.ground_truths.append(sample["label"])

        return True

    # ========================================================================
    # Calibration
    # ========================================================================

    def run_calibration(self) -> Dict:
        """Run calibration analysis on existing predictions."""
        self._init_components()
        tcfg = self.config.task

        if not self.predictions:
            self.load_results()
        if not self.predictions:
            raise RuntimeError("No predictions. Run pipeline first.")

        logger.info("Running calibration analysis...")
        calib_cfg = self.config.calibration

        ece_results = []
        corr_results = []

        for pred, unc, gt in zip(self.predictions, self.uncertainty_maps, self.ground_truths):
            # Error map: 1 where prediction != GT
            if self.task == "depth_estimation":
                error_map = np.abs(pred - gt) > 0.1  # threshold for depth
            else:
                error_map = (pred != gt) & (gt != tcfg.ignore_index)

            ece = compute_ece_from_uncertainty(
                unc, error_map,
                num_bins=calib_cfg.num_bins,
                strategy=calib_cfg.strategy,
            )
            ece_results.append(ece)

            corr = correlation_report(unc, error_map)
            corr_results.append(corr)

        # Average across images
        avg_ece = float(np.mean([r["ece"] for r in ece_results]))
        avg_mce = float(np.mean([r["mce"] for r in ece_results]))
        avg_spearman = float(np.mean([r["spearman_r"] for r in corr_results]))
        avg_auroc = float(np.mean([r["auroc"] for r in corr_results]))

        self.calibration_result = {
            "ece": avg_ece,
            "mce": avg_mce,
            "spearman_r": avg_spearman,
            "auroc": avg_auroc,
            "per_image_ece": ece_results,
            "per_image_correlation": corr_results,
        }

        logger.info(f"Calibration: ECE={avg_ece:.4f}, Spearman ρ={avg_spearman:.4f}, AUROC={avg_auroc:.4f}")

        # Plot calibration summary
        if ece_results and corr_results:
            plot_calibration_summary(
                ece_results[0],
                corr_results[0],
                output_path=str(self.output_dir / "calibration_summary.png"),
            )

        return self.calibration_result

    # ========================================================================
    # Robustness
    # ========================================================================

    def run_robustness(self, num_prompts: int = 50) -> Dict:
        """Run GVU-Robust benchmark."""
        self._init_components()

        from .benchmark.runner import GVURobustBenchmark
        benchmark = GVURobustBenchmark(
            config=self.config.robustness,
            output_dir=str(self.output_dir / "benchmark"),
        )

        # Generate prompt variants
        prompts = self.prompt_pool.generate_variants(n=num_prompts, seed=self.config.seed)

        # Build a simple inference function
        tcfg = self.config.task

        def inference_fn(image_pil, prompt, seed=None):
            if isinstance(image_pil, np.ndarray):
                image_pil = Image.fromarray(image_pil)
            # Use first available model
            model_name = list(self.sampler.generators.keys())[0]
            gen = self.sampler.generators[model_name]
            result = gen.generate(image_pil, prompt, seed=seed)
            mask = np.array(result)
            return self.decoder.decode(mask)

        def eval_fn(pred, gt):
            if self.task == "depth_estimation":
                from .metrics.metrics import depth_metrics
                return depth_metrics(pred, gt).get("delta1", 0.0)
            elif self.task == "referring_segmentation":
                from .metrics.metrics import cumulative_iou
                return cumulative_iou(pred, gt)
            else:
                return segmentation_metrics(pred, gt, tcfg.num_classes, tcfg.ignore_index)["mIoU"]

        # Prepare images
        images = []
        gts = []
        for idx in range(min(len(self.dataset), 20)):  # limit for speed
            sample = self.dataset[idx]
            img = sample["image"]
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            images.append(img)
            gts.append(sample["label"])

        # Build per-model inference fns for MRS
        model_fns = {}
        for mname, gen in self.sampler.generators.items():
            def _make_fn(g):
                return lambda img, prompt, seed=None: self.decoder.decode(np.array(g.generate(img, prompt, seed=seed)))
            model_fns[mname] = _make_fn(gen)

        self.robustness_result = benchmark.run(
            images=images,
            ground_truths=gts,
            prompts=prompts[:num_prompts],
            inference_fn=inference_fn,
            eval_fn=eval_fn,
            model_inference_fns=model_fns,
        )

        return self.robustness_result

    # ========================================================================
    # Persistence
    # ========================================================================

    def _save_results(self):
        """Save predictions, uncertainties, and metrics."""
        # Predictions and uncertainties are already saved incrementally in _process_single_sample.
        # Here we just save the aggregated metrics and config.

        # Save metrics
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(self.metrics_result, f, indent=2)

        # Save config
        self.config.to_yaml(self.output_dir / "config.yaml")

        logger.info(f"Results saved to {self.output_dir}")


