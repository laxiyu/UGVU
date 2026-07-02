"""UGVU 实验 — 真实街景照片

使用从网上下载的真实照片测试千问分割能力
"""
import sys, os, json, time, io, base64, re
sys.path.insert(0, '.')
os.environ['DASHSCOPE_API_KEY'] = 'sk-ws-H.RELRMHL.MTzq.MEUCIQCGvqtGIYN4v5mgTwyWr-2-GWFwCT6J2L3xndS063BcowIgfDlG2czYc2EDOpX9L4EUcOYZtVvMRzamcgiJd12WYVU'

import numpy as np
from PIL import Image

os.makedirs('outputs', exist_ok=True)

# ============================================================
# 1. 加载真实图片
# ============================================================
W, H = 256, 256

img_a = Image.open('outputs/real_a.png').convert('RGB')
img_b = Image.open('outputs/real_b.png').convert('RGB')

# 图片内容说明
scenes = [
    ('fjords', img_a, 'Fjord landscape with mountains, water, sky, boats, forest'),
    ('city_night', img_b, 'City skyline at night with buildings, lights, sky, roads'),
]

for name, img, desc in scenes:
    img.save(f'outputs/exp_real_{name}_input.png')
    print(f'📷 {name}: {desc} ({img.size})')

# ============================================================
# 2. 初始化生成器
# ============================================================
from ugvu.generators.qwen import QwenGenerator
from ugvu.decoders.decoder import ColormapDecoder
from ugvu.uncertainty.uncertainty_map import build_uncertainty_map
from ugvu.consensus.consensus_fusion import consensus_fusion
from ugvu.consensus.majority_vote import majority_vote
from ugvu.calibration.ece import compute_ece_from_uncertainty
from ugvu.calibration.correlation import correlation_report
from ugvu.metrics.metrics import mean_iou

# 自定义色图 (4 类通用)
CMAP = np.array([
    [255, 0, 0],     # 0: 天空
    [0, 0, 255],     # 1: 建筑/山
    [0, 255, 0],     # 2: 地面/水
    [255, 255, 0],   # 3: 物体/船灯
], dtype=np.uint8)

gen = QwenGenerator(
    api_endpoint='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ['DASHSCOPE_API_KEY'],
    model_version='qwen-vl-max',
    api_mode='chat', timeout_sec=120, image_size=(W, H),
)
gen.colormap = CMAP
decoder = ColormapDecoder(colormap=CMAP, num_classes=4)

K = 3

# ============================================================
# 3. 运行实验
# ============================================================
results = {}

for scene_name, img, desc in scenes:
    print(f'\n{"="*55}')
    print(f'📷 {scene_name}: {desc}')
    print(f'{"="*55}')

    prompt = f'Analyze this {desc}. Output a 16x16 grid JSON with class per cell: 0=sky, 1=building/mountain, 2=ground/water/road, 3=other objects. Format: {{"grid": [[...], ...]}}'

    masks = []
    times = []

    for i in range(K):
        t0 = time.time()
        m = gen.generate(img, prompt)
        elapsed = time.time() - t0
        times.append(elapsed)
        cm = decoder.decode(np.array(m.convert('RGB')))
        masks.append(cm)
        cls_dist = [np.sum(cm == c) for c in range(4)]
        print(f'  [{i+1}/{K}] {elapsed:.1f}s | 0={cls_dist[0]} 1={cls_dist[1]} 2={cls_dist[2]} 3={cls_dist[3]}')
        if i < K - 1:
            time.sleep(2)

    stack = np.stack(masks)

    # PCU
    unc = build_uncertainty_map(stack, num_classes=4)
    print(f'  PCU: mean={unc.mean():.4f} max={unc.max():.4f} H>0.5占比={np.sum(unc>0.5)/unc.size*100:.1f}%')

    # 可视化不确定度
    Image.fromarray((unc * 255).astype(np.uint8)).save(f'outputs/exp_real_{scene_name}_unc.png')

    # UGCF
    per_unc = np.stack([build_uncertainty_map(stack[i:i+1], num_classes=4) for i in range(K)])
    fused = consensus_fusion(stack, per_unc, method='ugcf', num_classes=4)
    fused_dist = [np.sum(fused == c) for c in range(4)]
    print(f'  UGCF:   0={fused_dist[0]} 1={fused_dist[1]} 2={fused_dist[2]} 3={fused_dist[3]}')

    # 可视化融合结果
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(4):
        vis[fused == c] = CMAP[c]
    Image.fromarray(vis).save(f'outputs/exp_real_{scene_name}_fused.png')

    # MVote
    mv = majority_vote(stack)
    mv_dist = [np.sum(mv == c) for c in range(4)]
    print(f'  MVote:  0={mv_dist[0]} 1={mv_dist[1]} 2={mv_dist[2]} 3={mv_dist[3]}')

    # Agreement
    from ugvu.uncertainty.variance import agreement_ratio
    agree = agreement_ratio(stack)
    print(f'  Agreement: mean={agree.mean():.4f}')

    # Calibration (以 MVote 为"伪 GT"做校准分析)
    error_map = (mv != fused) & (unc > 0.05)
    if error_map.sum() > 100:
        ece = compute_ece_from_uncertainty(unc, error_map, num_bins=10)
        corr = correlation_report(unc, error_map)
        print(f'  Calibration: ECE={ece["ece"]:.4f} Spearman ρ={corr["spearman_r"]:.4f} AUROC={corr["auroc"]:.4f}')
    else:
        print(f'  Calibration: 样本不足，跳过')

    # 模型响应示例（看原文）
    print(f'  ⏱ 均值={np.mean(times):.1f}s ± {np.std(times):.1f}s')

    results[scene_name] = {
        'pcu_mean': float(unc.mean()),
        'pcu_max': float(unc.max()),
        'agreement': float(agree.mean()),
        'fused_dist': [int(x) for x in fused_dist],
        'times': [float(t) for t in times],
    }

# ============================================================
# 4. 综合报告
# ============================================================
print(f'\n{"="*55}')
print(f'  📊 真实图片实验报告')
print(f'{"="*55}')

for name, r in results.items():
    print(f'\n{name}:')
    print(f'  PCU 不确定度: mean={r["pcu_mean"]:.4f} max={r["pcu_max"]:.4f}')
    print(f'  Agreement: {r["agreement"]:.4f}')
    print(f'  UGCF 分布: 天空={r["fused_dist"][0]} 建筑/山={r["fused_dist"][1]} 地面/水={r["fused_dist"][2]} 物体={r["fused_dist"][3]}')
    print(f'  API 耗时: {np.mean(r["times"]):.1f}s ± {np.std(r["times"]):.1f}s')

with open('outputs/exp_real_report.json', 'w') as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'times'} for k, v in results.items()}, f, indent=2)

print(f'\n✅ 保存在 outputs/ 目录')
print(f'   输入: exp_real_*_input.png')
print(f'   不确定度: exp_real_*_unc.png')
print(f'   融合 mask: exp_real_*_fused.png')
print(f'   报告: exp_real_report.json')
