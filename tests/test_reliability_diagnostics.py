import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def test_cross_model_fusion_uses_spatial_uncertainty():
    from ugvu.consensus.cross_model_vote import cross_model_fusion, cross_model_pixel_weights

    mask_a = np.zeros((4, 4), dtype=np.int64)
    mask_b = np.ones((4, 4), dtype=np.int64)
    unc_a = np.vstack([
        np.full((2, 4), 0.05, dtype=np.float32),
        np.full((2, 4), 0.95, dtype=np.float32),
    ])
    unc_b = np.vstack([
        np.full((2, 4), 0.95, dtype=np.float32),
        np.full((2, 4), 0.05, dtype=np.float32),
    ])

    weights = cross_model_pixel_weights({"a": unc_a, "b": unc_b}, temperature=0.25)
    assert np.all(weights["a"][:2] > weights["b"][:2])
    assert np.all(weights["b"][2:] > weights["a"][2:])

    fused = cross_model_fusion(
        {"a": mask_a, "b": mask_b},
        {"a": unc_a, "b": unc_b},
        num_classes=2,
        temperature=0.25,
    )
    assert np.all(fused[:2] == 0)
    assert np.all(fused[2:] == 1)


def test_uncertainty_decomposition_reports_within_and_between():
    from ugvu.diagnostics import uncertainty_decomposition

    gt = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.int64)
    model_a = np.stack([gt, gt, gt], axis=0)
    model_b = np.stack([1 - (gt == 1).astype(np.int64), gt, gt], axis=0)

    report = uncertainty_decomposition(
        {"a": model_a, "b": model_b},
        num_classes=3,
        ignore_index=255,
        lambda_within=0.6,
    )

    assert report["within_mean_map"].shape == gt.shape
    assert report["between_map"].shape == gt.shape
    assert report["combined_map"].shape == gt.shape
    assert report["summary"]["lambda_within"] == 0.6
    assert 0.0 <= report["summary"]["model_disagreement_rate"] <= 1.0
    assert report["summary"]["mean_within_uncertainty"] is not None

