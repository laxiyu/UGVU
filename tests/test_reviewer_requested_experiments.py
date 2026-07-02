
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_dirichlet_entropy_is_finite_and_less_extreme_at_small_k():
    from ugvu.uncertainty.entropy import pixelwise_dirichlet_entropy, pixelwise_entropy

    masks = np.array([
        [[0, 0, 1, 255]],
        [[0, 1, 1, 255]],
    ], dtype=np.int64)
    raw = pixelwise_entropy(masks, num_classes=3, ignore_index=255)
    smoothed = pixelwise_dirichlet_entropy(masks, num_classes=3, ignore_index=255, alpha=0.5)

    assert np.isfinite(smoothed).all()
    assert smoothed.min() >= 0.0 and smoothed.max() <= 1.0
    assert smoothed[0, 0] > raw[0, 0]
    assert smoothed[0, 3] == 0.0


def test_uncertainty_map_builder_supports_dirichlet_entropy():
    from ugvu.uncertainty.uncertainty_map import build_uncertainty_map

    masks = np.stack([
        np.array([[0, 1], [1, 2]], dtype=np.int64),
        np.array([[0, 1], [2, 2]], dtype=np.int64),
        np.array([[0, 2], [2, 2]], dtype=np.int64),
    ])
    unc = build_uncertainty_map(masks, uncertainty_type="dirichlet_entropy", num_classes=3, alpha=0.5)
    assert unc.shape == (2, 2)
    assert float(unc[0, 0]) < float(unc[0, 1])


def test_failure_tags_fire_on_formal_synthetic_cases():
    from ugvu.diagnostics import prediction_diagnostics

    gt = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
    ], dtype=np.int64)
    collapsed = np.zeros_like(gt)
    unc_bad = np.zeros_like(gt, dtype=np.float32)
    diag = prediction_diagnostics(collapsed, gt, unc_bad, ignore_index=255, num_classes=5)

    assert "class_collapse" in diag["failure_tags"]
    assert "low_class_diversity" in diag["failure_tags"]
    assert "high_error_rate" in diag["failure_tags"]
    assert "uncertainty_misaligned" in diag["failure_tags"]


def test_decoder_sensitivity_changes_invalid_rate():
    from ugvu.decoders.decoder import ColormapDecoder

    cmap = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)
    image = np.array([
        [[255, 0, 0], [250, 5, 0], [120, 120, 120]],
    ], dtype=np.uint8)

    strict = ColormapDecoder(cmap, num_classes=2, ignore_index=255, max_distance=4.0)
    permissive = ColormapDecoder(cmap, num_classes=2, ignore_index=255, max_distance=220.0)
    strict_mask, strict_unc = strict.decode_with_uncertainty(image)
    permissive_mask, permissive_unc = permissive.decode_with_uncertainty(image)

    assert int((strict_mask == 255).sum()) > int((permissive_mask == 255).sum())
    assert float(strict_unc.mean()) > float(permissive_unc.mean())


def test_ugcf_and_cmcf_improve_constructed_complementary_case():
    from ugvu.consensus.majority_vote import majority_vote
    from ugvu.consensus.weighted_vote import uncertainty_guided_fusion
    from ugvu.consensus.cross_model_vote import cross_model_fusion

    gt = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int64)
    stack = np.stack([
        1 - gt,
        gt,
        1 - gt,
    ])
    uncs = np.stack([
        np.full(gt.shape, 0.9, dtype=np.float32),
        np.full(gt.shape, 0.05, dtype=np.float32),
        np.full(gt.shape, 0.8, dtype=np.float32),
    ])

    mv = majority_vote(stack)
    ugcf = uncertainty_guided_fusion(stack, uncs, num_classes=2, temperature=0.25)
    assert float((ugcf == gt).mean()) > float((mv == gt).mean())

    model_a = np.zeros_like(gt)
    model_b = np.ones_like(gt)
    unc_a = np.where(gt == 0, 0.05, 0.9).astype(np.float32)
    unc_b = np.where(gt == 1, 0.05, 0.9).astype(np.float32)
    cmcf = cross_model_fusion({"a": model_a, "b": model_b}, {"a": unc_a, "b": unc_b}, num_classes=2, temperature=0.25)
    assert np.array_equal(cmcf, gt)
