from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ugvu.configs.config import UGVUConfig
from ugvu.datasets.dataset import ADE20K_CLASSES, CITYSCAPES_CLASSES, CITYSCAPES_COLORMAP, build_dataset
from ugvu.generators.sampler import build_sampler
from ugvu.uncertainty.uncertainty_map import build_uncertainty_map
from ugvu.consensus.consensus_fusion import consensus_fusion
from ugvu.metrics.metrics import evaluate_dataset, segmentation_metrics
from ugvu.calibration.ece import compute_ece_from_uncertainty
from ugvu.calibration.correlation import correlation_report
from ugvu.visualization.visualize import save_uncertainty_map
from ugvu.diagnostics import aggregate_failure_tags, attach_k_sweep_summary, prediction_diagnostics, semantic_trivial_baselines

CITY_ROOT = r'D:\BaiduNetdiskDownload\cityscapes'
ADE_ROOT = r'D:\BaiduNetdiskDownload\ADE20K'
RIGHT_CODE_MODELS = ('gpt-image-2-vip', 'gpt-image-2', 'nano-banana', 'nano-banana-2')


def right_code_model(model_version: str, image_size=(1024, 1024)) -> dict:
    return {
        'api_endpoint': 'https://www.right.codes/draw/v1',
        'api_key': '${IMAGE_API_KEY}',
        'model_version': model_version,
        'api_mode': 'chat_completions',
        'class': 'rightcode',
        'timeout_sec': 300,
        'max_retries': 2,
        'image_size': list(image_size),
        'size': '1K',
        'temperature': 0.2,
    }


def model_registry(dataset: str) -> dict:
    image_size = (1024, 512) if dataset == 'cityscapes' else (1024, 1024)
    models = {
        'qwen-vl': {
            'api_endpoint': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
            'api_key': '${DASHSCOPE_API_KEY}',
            'model_version': 'qwen-vl-max',
            'api_mode': 'chat',
            'timeout_sec': 120,
            'max_retries': 1,
            'image_size': list(image_size),
            'temperature': 0.2,
        },
        'doubao': {
            'api_endpoint': 'https://ark.cn-beijing.volces.com/api/v3',
            'api_key': '${DOUBAO_API_KEY}',
            'model_version': 'ep-m-20260616184522-wp9rq',
            'api_mode': 'vision_chat',
            'timeout_sec': 120,
            'max_retries': 1,
            'image_size': list(image_size),
            'temperature': 0.2,
        },
    }
    for name in RIGHT_CODE_MODELS:
        models[name] = right_code_model(name, image_size=(1024, 1024))
    return models


def load_keys() -> dict:
    cfg = UGVUConfig.from_yaml('configs/default.yaml')
    return {name: model.api_key for name, model in cfg.models.models.items()}


def city_prompt(candidates: list[int]) -> str:
    spec = '; '.join(f'{i}={CITYSCAPES_CLASSES[c]} (Cityscapes train id {c})' for i, c in enumerate(candidates))
    return (
        'Perform Cityscapes semantic segmentation using ONLY the candidate classes below. '
        'Output local candidate IDs, not color names. Candidate local IDs: ' + spec + '. '
        'Every grid value must be one of these local IDs. Return dense class-index grid only; no 255, no unknown, no text labels. '
        'Use coherent regions and preserve road, building, sky, vegetation, person, and vehicle boundaries.'
    )


def ade_prompt(candidates: list[int]) -> str:
    spec = '; '.join(f'{i}={ADE20K_CLASSES[c]} (ADE id {c})' for i, c in enumerate(candidates))
    return (
        'Perform ADE20K semantic segmentation using ONLY the candidate classes below. '
        'Output local candidate IDs, not ADE ids. Candidate local IDs: ' + spec + '. '
        'Every grid value must be one of these local IDs. Return dense class-index grid only; no 255, no unknown, no text labels. '
        'Use coherent regions and preserve major object boundaries.'
    )


def decode_candidate_stack(stack: np.ndarray, candidates: list[int]) -> tuple[np.ndarray, np.ndarray]:
    local_stack, unc_stack = [], []
    for generated in stack:
        values = (generated[..., 0] if generated.ndim == 3 else generated).astype(np.float32)
        nearest = np.rint(values).astype(np.int64)
        local = np.full(nearest.shape, 255, dtype=np.int64)
        for local_id, raw_id in enumerate(candidates):
            local[nearest == local_id] = local_id
            local[nearest == int(raw_id)] = local_id
        unc = np.clip(np.abs(values - np.rint(values)), 0, 1).astype(np.float32)
        unc[local == 255] = 1.0
        local_stack.append(local)
        unc_stack.append(unc)
    return np.stack(local_stack, axis=0), np.stack(unc_stack, axis=0)


def map_local(local_pred: np.ndarray, candidates: list[int]) -> np.ndarray:
    out = np.full(local_pred.shape, 255, dtype=np.int64)
    for local_id, raw_id in enumerate(candidates):
        out[local_pred == local_id] = raw_id
    return out


def save_city_fig(image, pred, gt, unc, path: Path):
    def color(mask):
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for c in range(len(CITYSCAPES_COLORMAP)):
            rgb[mask == c] = CITYSCAPES_COLORMAP[c]
        return rgb
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image); axes[0].set_title('Input'); axes[0].axis('off')
    axes[1].imshow(color(pred)); axes[1].set_title('Prediction'); axes[1].axis('off')
    axes[2].imshow(color(gt)); axes[2].set_title('GT'); axes[2].axis('off')
    im = axes[3].imshow(unc, cmap='hot', vmin=0, vmax=1); axes[3].set_title('Uncertainty'); axes[3].axis('off')
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def save_ade_fig(image, pred, gt, unc, path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image); axes[0].set_title('Input'); axes[0].axis('off')
    axes[1].imshow(pred, cmap='tab20', vmin=0, vmax=149); axes[1].set_title('Prediction'); axes[1].axis('off')
    axes[2].imshow(gt, cmap='tab20', vmin=0, vmax=149); axes[2].set_title('GT'); axes[2].axis('off')
    im = axes[3].imshow(unc, cmap='hot', vmin=0, vmax=1); axes[3].set_title('Uncertainty'); axes[3].axis('off')
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def run_case(dataset_name: str, model_name: str, k: int, dataset, key: str, out_root: Path) -> dict:
    out_dir = out_root / f'{model_name}_k{k}'
    out_dir.mkdir(parents=True, exist_ok=True)
    models = model_registry(dataset_name)
    model_cfg = models[model_name].copy(); model_cfg['api_key'] = key
    sampler = build_sampler({model_name: model_cfg}, k_samples=k, parallel=False, max_concurrent=1, seed=42, cache_dir=str(out_dir / 'cache'))
    num_classes = 19 if dataset_name == 'cityscapes' else 150
    prompt_fn = city_prompt if dataset_name == 'cityscapes' else ade_prompt
    save_fig = save_city_fig if dataset_name == 'cityscapes' else save_ade_fig
    predictions, gts, uncs, per, failures = [], [], [], [], []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        image_arr = sample['image']; gt = sample['label'].astype(np.int64)
        pred_path = out_dir / f'pred_{idx:04d}.npy'; unc_path = out_dir / f'unc_{idx:04d}.npy'
        candidates = sorted(int(v) for v in np.unique(gt) if int(v) != 255)
        if pred_path.exists() and unc_path.exists():
            pred = np.load(pred_path); unc = np.load(unc_path)
        else:
            try:
                collection = sampler.sample(Image.fromarray(image_arr), prompts=[prompt_fn(candidates)] * k, models=[model_name], image_id=f'{dataset_name}_{idx:04d}')
                local_stack, dec_unc = decode_candidate_stack(collection.mask_stack, candidates)
                unc = build_uncertainty_map(local_stack, task='semantic_segmentation', uncertainty_type='entropy', num_classes=len(candidates), ignore_index=255)
                unc = np.maximum(unc, dec_unc.mean(axis=0))
                per_unc = np.maximum(dec_unc, np.tile(unc[None, ...], (local_stack.shape[0], 1, 1)))
                local_pred = consensus_fusion(local_stack, per_unc, method='majority' if k == 1 else 'ugcf', num_classes=len(candidates), ignore_index=255, temperature=0.5)
                pred = map_local(local_pred, candidates)
                if pred.shape[:2] != gt.shape[:2]:
                    pred = np.asarray(Image.fromarray(pred.astype(np.int32)).resize((gt.shape[1], gt.shape[0]), resample=Image.NEAREST), dtype=np.int64)
                    unc = np.asarray(Image.fromarray(unc.astype(np.float32)).resize((gt.shape[1], gt.shape[0]), resample=Image.BILINEAR), dtype=np.float32)
                np.save(pred_path, pred); np.save(unc_path, unc); save_uncertainty_map(unc, str(out_dir / f'unc_{idx:04d}.png')); save_fig(image_arr, pred, gt, unc, out_dir / f'comparison_{idx:04d}.png')
            except Exception as exc:
                failures.append({'image': idx, 'error': str(exc)})
                continue
        predictions.append(pred); gts.append(gt); uncs.append(unc)
        vals, cnts = np.unique(pred[pred != 255], return_counts=True)
        class_names = CITYSCAPES_CLASSES if dataset_name == 'cityscapes' else ADE20K_CLASSES
        per.append({'image': idx, 'pred_classes': [{'id': int(v), 'name': class_names[int(v)] if 0 <= int(v) < len(class_names) else str(v), 'pixels': int(n)} for v, n in zip(vals, cnts)]})

    result = {'dataset': dataset_name, 'protocol': 'oracle_candidates', 'model': model_name, 'k': k, 'max_samples': len(dataset), 'completed_samples': len(predictions), 'failed_samples': len(failures), 'failures': failures, 'ok': bool(predictions), 'metrics': {}, 'per_image': per, 'diagnostics': {}}
    if not predictions:
        return result
    if dataset_name == 'cityscapes':
        metrics = evaluate_dataset(predictions, gts, task='semantic_segmentation', num_classes=num_classes, ignore_index=255)
        miou = metrics.get('mIoU')
    else:
        per_metrics = [segmentation_metrics(p, g, num_classes=num_classes, ignore_index=255) for p, g in zip(predictions, gts)]
        miou = float(np.mean([m['mIoU'] for m in per_metrics]))
    eces, corrs = [], []
    for pred, gt, unc in zip(predictions, gts, uncs):
        err = (pred != gt) & (gt != 255)
        eces.append(compute_ece_from_uncertainty(unc, err, num_bins=15, strategy='uniform'))
        corrs.append(correlation_report(unc, err))
    diag_per = [prediction_diagnostics(pred, gt, unc, ignore_index=255, num_classes=num_classes) for pred, gt, unc in zip(predictions, gts, uncs)]
    result['metrics'] = {'mIoU': miou, 'ECE': float(np.mean([e['ece'] for e in eces])), 'Spearman': float(np.nanmean([c['spearman_r'] for c in corrs])), 'AUROC': float(np.nanmean([c['auroc'] for c in corrs]))}
    result['diagnostics'] = {'per_image': diag_per, 'failure_tag_counts': aggregate_failure_tags(diag_per), 'trivial_baselines': semantic_trivial_baselines(gts, num_classes=num_classes, ignore_index=255)}
    return result


def run_suite(dataset_name: str, models: list[str], k_values: list[int], max_samples: int, output_dir: str):
    root = CITY_ROOT if dataset_name == 'cityscapes' else ADE_ROOT
    dataset = build_dataset(dataset_name, root=root, split='val', max_samples=max_samples)
    keys = load_keys(); out_root = Path(output_dir); out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for model in models:
        for k in k_values:
            key_name = model
            result = run_case(dataset_name, model, k, dataset, keys[key_name], out_root)
            results.append(result)
            with (out_root / f'result_{model}_k{k}.json').open('w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            with (out_root / 'partial_results.json').open('w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    report = {'dataset': dataset_name, 'protocol': 'oracle_candidates', 'max_samples': max_samples, 'models': models, 'k_values': k_values, 'k_sweep_summary': attach_k_sweep_summary(results), 'results': results}
    with (out_root / 'candidate_diagnostic_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cityscapes', 'ade20k'], required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--k-values', required=True)
    parser.add_argument('--max-samples', type=int, default=20)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    k_values = [int(v.strip()) for v in args.k_values.split(',') if v.strip()]
    run_suite(args.dataset, models, k_values, args.max_samples, args.output_dir)


if __name__ == '__main__':
    main()
