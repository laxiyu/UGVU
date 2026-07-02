"""核心算法模块的单元测试 — 无需 API Key, 纯 CPU 可跑."""

import sys
sys.path.insert(0, '.')
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from scipy import stats


# ============================================================================
# Test 1: 不确定性 (PCU)
# ============================================================================

def test_pixelwise_entropy():
    from ugvu.uncertainty.entropy import pixelwise_entropy

    # 5 张 mask, 3 类, 4x4
    masks = np.array([
        [[0, 0, 1, 1],
         [0, 0, 1, 1],
         [2, 2, 0, 0],
         [2, 2, 0, 0]],
    ] * 5)  # (5, 4, 4) — 所有样本一致

    unc = pixelwise_entropy(masks, num_classes=3)
    # 全一致 → 熵应为 0
    assert unc.max() < 1e-6, f"Expected 0 entropy for unanimous, got {unc.max()}"
    print("  ✅ test_pixelwise_entropy: 全一致 → H=0")

    # 完全随机: 每个像素 5 张全部不同类
    rng = np.random.RandomState(42)
    masks_random = rng.randint(0, 3, size=(5, 4, 4))
    unc_random = pixelwise_entropy(masks_random, num_classes=3)
    # 熵应该 > 0
    assert unc_random.mean() > 0, "Random should have positive entropy"
    print(f"  ✅ test_pixelwise_entropy: 随机 → H_mean={unc_random.mean():.4f}")

    # 部分不一致
    masks_mixed = np.zeros((5, 2, 2), dtype=np.int64)
    masks_mixed[:3, 0, 0] = 0   # 3/5 是 0
    masks_mixed[3:, 0, 0] = 1   # 2/5 是 1
    unc_mixed = pixelwise_entropy(masks_mixed, num_classes=3)
    # (0,0) 像素应有中等熵
    assert unc_mixed[0, 0] > 0, "Mixed pixel should have entropy > 0"
    assert unc_mixed[1, 1] == 0, "Unanimous pixel should have entropy = 0"
    print(f"  ✅ test_pixelwise_entropy: 混合 → H(0,0)={unc_mixed[0,0]:.4f}")


def test_uncertainty_map_builder():
    from ugvu.uncertainty.uncertainty_map import build_uncertainty_map

    masks = np.random.RandomState(0).randint(0, 5, size=(5, 8, 8))
    unc = build_uncertainty_map(masks, task="semantic_segmentation",
                                uncertainty_type="entropy", num_classes=5)
    assert unc.shape == (8, 8), f"Expected (8,8), got {unc.shape}"
    assert unc.dtype == np.float32
    assert 0 <= unc.min() and unc.max() <= 1.0, "Entropy should be normalized [0,1]"
    print(f"  ✅ test_uncertainty_map_builder: shape={unc.shape}, range=[{unc.min():.3f}, {unc.max():.3f}]")

    # Agreement mode
    unc_agree = build_uncertainty_map(masks, uncertainty_type="agreement", num_classes=5)
    assert unc_agree.shape == (8, 8)
    print(f"  ✅ test_uncertainty_map_builder (agreement): OK")


# ============================================================================
# Test 2: 共识融合 (Consensus)
# ============================================================================

def test_majority_vote():
    from ugvu.consensus.majority_vote import majority_vote

    masks = np.array([
        [[0, 0], [0, 0]],
        [[1, 0], [1, 0]],
        [[1, 0], [1, 0]],
    ])  # (3, 2, 2) — 位置 (0,0): 0,1,1 → 1
    result = majority_vote(masks)
    assert result[0, 0] == 1, f"Expected 1 at (0,0), got {result[0,0]}"
    assert result[0, 1] == 0, f"Expected 0 at (0,1), got {result[0,1]}"
    print("  ✅ test_majority_vote: OK")


def test_ugcf():
    from ugvu.consensus.weighted_vote import uncertainty_guided_fusion

    masks = np.zeros((3, 4, 4), dtype=np.int64)
    masks[0] = 0
    masks[1] = 1
    masks[2] = 1
    # 不确定性: 第 0 张高不确定 → 降权
    uncertainties = np.ones((3, 4, 4), dtype=np.float32)
    uncertainties[0] = 0.9   # 高不确定
    uncertainties[1] = 0.1   # 低不确定
    uncertainties[2] = 0.1

    result = uncertainty_guided_fusion(masks, uncertainties, num_classes=3, temperature=0.5)
    assert (result == 1).all(), "UGCF should favor low-uncertainty samples"
    print("  ✅ test_ugcf: 低不确定样本主导")


def test_cmcf():
    from ugvu.consensus.cross_model_vote import cross_model_fusion

    masks_a = np.zeros((4, 4), dtype=np.int64)
    masks_b = np.ones((4, 4), dtype=np.int64)
    unc_a = np.full((4, 4), 0.9, dtype=np.float32)  # 高不确定
    unc_b = np.full((4, 4), 0.1, dtype=np.float32)  # 低不确定

    result = cross_model_fusion(
        {"model_a": masks_a, "model_b": masks_b},
        {"model_a": unc_a, "model_b": unc_b},
        num_classes=3
    )
    assert (result == 1).all(), "CMCF should weight low-uncertainty model more"
    print("  ✅ test_cmcf: 低不确定模型主导")


# ============================================================================
# Test 3: 校准 (Calibration)
# ============================================================================

def test_ece():
    from ugvu.calibration.ece import compute_ece

    # 完美校准: confidence == accuracy
    conf = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    acc = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    result = compute_ece(conf, acc, num_bins=5)
    assert result["ece"] < 1e-6, f"Perfect calibration should have ECE=0, got {result['ece']}"
    print(f"  ✅ test_ece: 完美校准 ECE={result['ece']:.6f}")

    # 完全不校准
    conf_bad = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
    acc_bad = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    result_bad = compute_ece(conf_bad, acc_bad, num_bins=3)
    assert result_bad["ece"] > 0.5, f"Bad calibration should have high ECE"
    print(f"  ✅ test_ece: 不校准 ECE={result_bad['ece']:.4f}")


def test_correlation():
    from ugvu.calibration.correlation import spearman_correlation

    # 完美正相关: 高不确定 ↔ 高错误
    unc = np.linspace(0, 1, 100).reshape(10, 10)
    err = np.linspace(0, 1, 100).reshape(10, 10) > 0.5

    rho = spearman_correlation(unc, err.astype(np.int64))
    assert rho > 0.5, f"Expected strong positive correlation, got {rho}"
    print(f"  ✅ test_correlation: Spearman ρ={rho:.4f}")


# ============================================================================
# Test 4: 鲁棒性 (Robustness)
# ============================================================================

def test_prs():
    from ugvu.robustness.prompt_variance import prompt_robustness_score

    # 完全稳定 → PRS = 1.0
    metrics = np.ones(10) * 0.8
    prs = prompt_robustness_score(metrics)
    assert abs(prs - 1.0) < 1e-6, f"Perfect stability should have PRS=1, got {prs}"
    print(f"  ✅ test_prs: 完全稳定 PRS={prs:.4f}")

    # 有波动
    metrics_var = np.array([0.7, 0.8, 0.75, 0.72, 0.78])
    prs_var = prompt_robustness_score(metrics_var)
    assert prs_var < 1.0, "Variable metrics should reduce PRS"
    print(f"  ✅ test_prs: 有波动 PRS={prs_var:.4f}")


# ============================================================================
# Test 5: 指标 (Metrics)
# ============================================================================

def test_miou():
    from ugvu.metrics.metrics import mean_iou, confusion_matrix

    pred = np.array([[0, 1], [2, 0]], dtype=np.int64)
    target = np.array([[0, 1], [2, 0]], dtype=np.int64)
    miou = mean_iou(pred, target, num_classes=3)
    assert abs(miou - 1.0) < 1e-6, f"Perfect match should have mIoU=1, got {miou}"
    print(f"  ✅ test_miou: 完全匹配 mIoU={miou:.4f}")

    # 完全不匹配
    pred_bad = np.zeros((4, 4), dtype=np.int64)
    target_bad = np.ones((4, 4), dtype=np.int64)
    miou_bad = mean_iou(pred_bad, target_bad, num_classes=3)
    assert miou_bad < 0.1
    print(f"  ✅ test_miou: 完全不匹配 mIoU={miou_bad:.4f}")


def test_depth_metrics():
    from ugvu.metrics.metrics import depth_metrics

    pred = np.ones((10, 10), dtype=np.float32) * 5.0
    target = np.ones((10, 10), dtype=np.float32) * 5.0
    d = depth_metrics(pred, target, median_scale=False)
    assert d["AbsRel"] < 1e-6, f"Perfect depth should have AbsRel=0, got {d['AbsRel']}"
    assert d["delta1"] > 0.999
    print(f"  ✅ test_depth_metrics: 完美深度 AbsRel={d['AbsRel']:.4f}, δ1={d['delta1']:.4f}")


# ============================================================================
# Test 6: 解码器 (Decoder)
# ============================================================================

def test_colormap_decoder():
    from ugvu.decoders.decoder import ColormapDecoder
    import numpy as np

    cmap = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    decoder = ColormapDecoder(colormap=cmap, num_classes=3)

    # 生成一个红色图像 → 应解码为 class 0
    img = np.ones((4, 4, 3), dtype=np.uint8) * np.array([255, 0, 0], dtype=np.uint8)
    mask = decoder.decode(img)
    assert (mask == 0).all(), f"Red should decode to class 0, got {np.unique(mask)}"
    print(f"  ✅ test_colormap_decoder: 红色→类0 OK")

    # 绿色 → class 1
    img_green = np.ones((4, 4, 3), dtype=np.uint8) * np.array([0, 255, 0], dtype=np.uint8)
    mask_green = decoder.decode(img_green)
    assert (mask_green == 1).all()
    print(f"  ✅ test_colormap_decoder: 绿色→类1 OK")

    invalid = np.ones((4, 4, 3), dtype=np.uint8) * np.array([128, 128, 128], dtype=np.uint8)
    strict_decoder = ColormapDecoder(
        colormap=cmap,
        num_classes=3,
        ignore_index=255,
        max_distance=20.0,
    )
    mask_invalid, unc_invalid = strict_decoder.decode_with_uncertainty(invalid)
    assert (mask_invalid == 255).all()
    assert unc_invalid.min() == 1.0
    print("  ✅ test_colormap_decoder: OOD 颜色拒绝 OK")


# ============================================================================
# Test 7: RLE 编解码 (Qwen 文本解析)
# ============================================================================

def test_rle():
    from ugvu.generators.qwen import _rle_encode, _rle_decode
    import numpy as np

    mask = np.zeros((8, 8), dtype=np.int64)
    mask[2:6, 2:6] = 1
    rle = _rle_encode(mask)
    decoded = _rle_decode(rle, 8, 8)
    assert (decoded == mask).all(), "RLE roundtrip failed"
    print(f"  ✅ test_rle: 编解码一致 ({len(rle)} chars)")


def test_json_mask_parsing():
    from ugvu.generators.qwen import _parse_json_mask
    import numpy as np

    # Grid format
    text = '```json\n{"grid": [[0,1,0],[1,1,0],[0,0,1]]}\n```'
    mask = _parse_json_mask(text, 6, 6)
    assert mask is not None, "Grid parsing should succeed"
    assert mask.shape == (6, 6)
    print(f"  ✅ test_json_mask_parsing: Grid 解析 OK")

    # RLE format
    text2 = '{"rle": "64 64"}'
    mask2 = _parse_json_mask(text2, 8, 16)
    assert mask2 is not None
    print(f"  ✅ test_json_mask_parsing: RLE 解析 OK")


# ============================================================================
# Test 8: 提示词池 (Prompt Pool)
# ============================================================================

def test_prompt_pool():
    from ugvu.prompts.prompt_pool import get_prompt_pool, generate_prompt_variants

    pool = get_prompt_pool("semantic_segmentation")
    assert len(pool.base_templates) >= 5
    print(f"  ✅ test_prompt_pool: {len(pool.base_templates)} 个基础模板")

    variants = generate_prompt_variants(pool.base_templates, n=20, seed=42)
    assert len(variants) == 20
    # 变体应该与基础模板不同
    assert any(len(set(v.split())) > 3 for v in variants)
    print(f"  ✅ test_prompt_pool: 生成了 {len(variants)} 个变体")


# ============================================================================
# Test 9: MockGenerator + Sampler 联调
# ============================================================================

def test_mock_pipeline():
    from ugvu.generators.base_generator import MockGenerator
    from ugvu.generators.sampler import Sampler
    from PIL import Image
    import numpy as np

    gen = MockGenerator(model_name="mock", noise_std=0.05)
    sampler = Sampler(generators={"mock": gen}, k_samples=3)

    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    prompts = ["test prompt"] * 3
    collection = sampler.sample(img, prompts)

    assert collection.k == 3, f"Expected 3 samples, got {collection.k}"
    assert len(collection.masks) == 3
    print(f"  ✅ test_mock_pipeline: Sampler 生成 {collection.k} 个样本 OK")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    tests = [
        ("不确定性 - 逐像素熵", test_pixelwise_entropy),
        ("不确定性 - 构建器", test_uncertainty_map_builder),
        ("共识 - 多数投票", test_majority_vote),
        ("共识 - UGCF", test_ugcf),
        ("共识 - CMCF", test_cmcf),
        ("校准 - ECE", test_ece),
        ("校准 - 相关性", test_correlation),
        ("鲁棒性 - PRS", test_prs),
        ("指标 - mIoU", test_miou),
        ("指标 - 深度", test_depth_metrics),
        ("解码器 - 色图", test_colormap_decoder),
        ("Qwen - RLE 编解码", test_rle),
        ("Qwen - JSON mask 解析", test_json_mask_parsing),
        ("提示词池", test_prompt_pool),
        ("端到端 - MockPipeline", test_mock_pipeline),
    ]

    passed = 0
    failed = 0
    print(f"\n{'='*60}")
    print(f"  UGVU 单元测试套件")
    print(f"{'='*60}\n")

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"  结果: {passed} 通过, {failed} 失败 / {len(tests)} 项")
    print(f"{'='*60}")
