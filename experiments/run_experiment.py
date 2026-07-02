"""千问 UGVU 实验 — 端到端验证

测试 2 张合成图 × K=3 次生成 → PCU → UGCF → Calibration
"""
import sys, os, json, time, io, base64
sys.path.insert(0, '.')
os.environ['DASHSCOPE_API_KEY'] = 'sk-ws-H.RELRMHL.MTzq.MEUCIQCGvqtGIYN4v5mgTwyWr-2-GWFwCT6J2L3xndS063BcowIgfDlG2czYc2EDOpX9L4EUcOYZtVvMRzamcgiJd12WYVU'

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ugvu.generators.qwen import QwenGenerator
from ugvu.decoders.decoder import ColormapDecoder
from ugvu.uncertainty.uncertainty_map import build_uncertainty_map
from ugvu.consensus.consensus_fusion import consensus_fusion
from ugvu.calibration.ece import compute_ece_from_uncertainty
from ugvu.calibration.correlation import correlation_report

CMAP = np.array([
    [255, 0, 0],     # 0: 天空
    [0, 0, 255],     # 1: 建筑
    [0, 255, 0],     # 2: 道路
    [255, 255, 0],   # 3: 车辆
], dtype=np.uint8)

CLASS_NAMES = ['天空', '建筑', '道路', '车辆']
K = 3

# ============================================================
# 1. 创建 2 张更真实的街景测试图 (用 PIL 绘图)
# ============================================================
W, H = 256, 256


def draw_street_scene(draw, has_second_car=True, seed=0):
    """用 PIL ImageDraw 绘制较真实的街景 (天空渐变 + 建筑群 + 道路 + 车辆 + 树)."""
    import random
    rng = random.Random(seed)
    rnd = lambda a, b: rng.randint(a, b)

    # --- 天空渐变 ---
    sky_h = int(H * 0.38)
    for y in range(sky_h):
        t = y / sky_h
        r = int(180 - t * 60)
        g = int(210 - t * 50)
        b = int(245 - t * 30)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # --- 远山/云层 ---
    for y in range(int(H * 0.08), int(H * 0.25)):
        t = (y - H * 0.08) / (H * 0.17)
        r = int(200 - t * 50)
        g = int(220 - t * 40)
        b = int(240 - t * 30)
        draw.line([(0, y), (W, y)], fill=(r, g, b))

    # --- 建筑群 ---
    skyline = sky_h
    draw.rectangle([0, skyline, W, int(H * 0.72)], fill=(155, 148, 140))

    bx = 0
    while bx < W:
        bw = rnd(20, 40)
        bh = rnd(18, 50)
        by = skyline - bh + rnd(-8, 8)
        bri = rnd(125, 195)
        color = (bri, bri - 3, bri - 8)
        draw.rectangle([bx, by, bx + bw, int(H * 0.72)], fill=color)
        # 窗户 (带随机亮灯)
        for wy in range(max(by + 3, skyline + 3), int(H * 0.72) - 3, 9):
            for wx in range(bx + 3, bx + bw - 3, 9):
                if rnd(0, 4) > 0:
                    wb = rnd(160, 255)
                    draw.rectangle([wx, wy, wx + 4, wy + 4], fill=(wb, wb, wb - 50))
        bx += bw + rnd(2, 6)

    # --- 道路 ---
    road_y = int(H * 0.72)
    draw.rectangle([0, road_y, W, H], fill=(62, 59, 55))
    # 车道线
    for ly in range(road_y + 12, H - 5, 20):
        for lx in range(6, W, 32):
            draw.rectangle([lx, ly, lx + 14, ly + 3], fill=(245, 235, 140))
    # 路沿
    draw.rectangle([0, road_y, W, road_y + 4], fill=(95, 88, 82))
    # 人行道纹理
    for sx in range(0, W, 8):
        for sy in range(road_y + 4, int(H * 0.74), 2):
            draw.point((sx, sy), fill=(75, 70, 65))

    # --- 车辆 ---
    def draw_car(x, y, color, w=42):
        h = 20
        draw.ellipse([x - 2, y + h - 4, x + w + 2, y + h + 4], fill=(30, 28, 25))
        draw.rectangle([x, y + 4, x + w, y + h], fill=color)
        draw.ellipse([x - 2, y + 6, x + 4, y + h - 2], fill=color)
        draw.ellipse([x + w - 4, y + 6, x + w + 2, y + h - 2], fill=color)
        rw = w - 12
        rx = x + 6
        draw.rectangle([rx, y, rx + rw, y + 8], fill=(max(0,color[0]-40), max(0,color[1]-40), max(0,color[2]-40)))
        glass = (170, 195, 230)
        draw.rectangle([rx + 2, y + 1, rx + rw//2 - 1, y + 6], fill=glass)
        draw.rectangle([rx + rw//2 + 1, y + 1, rx + rw - 2, y + 6], fill=glass)
        draw.ellipse([x + 5, y + h - 3, x + 11, y + h + 4], fill=(25, 25, 25))
        draw.ellipse([x + w - 11, y + h - 3, x + w - 5, y + h + 4], fill=(25, 25, 25))
        draw.rectangle([x, y + h - 6, x + 2, y + h - 2], fill=(255, 220, 80))

    cy = road_y + 8
    draw_car(20, cy + rnd(3, 12), (190, 45, 45), w=rnd(38, 46))
    if has_second_car:
        draw_car(100, cy + rnd(10, 22), (45, 85, 195), w=rnd(40, 48))
        draw_car(180, cy + rnd(0, 8), (50, 160, 70), w=rnd(35, 42))

    # --- 行道树 ---
    for tx in [12, 75, 145, 215, 250]:
        if tx > W - 5:
            continue
        ty = road_y - rnd(5, 18)
        draw.rectangle([tx, ty, tx + 5, road_y], fill=(70, 50, 35))
        cr = rnd(8, 14)
        cg = rnd(50, 110)
        draw.ellipse([tx - cr, ty - cr - 4, tx + cr, ty + cr - 4], fill=(25, cg, 35))
        draw.ellipse([tx - cr//2, ty - cr//2 - 4, tx + cr//4, ty + cr//4 - 4], fill=(60, cg+30, 60))


# ---- 图 A: 单车道场景 ----
img_a = Image.new('RGB', (W, H))
draw_a = ImageDraw.Draw(img_a)
draw_street_scene(draw_a, has_second_car=False, seed=1)
img_a_np = np.array(img_a)

# GT: 粗略像素级标注 (允许 ±2px 容差)
gt_a = np.zeros((H, W), dtype=np.int64)
gt_a[:int(H * 0.38)] = 0
gt_a[int(H * 0.38):int(H * 0.72)] = 1
gt_a[int(H * 0.72):] = 2
gt_a[int(H * 0.72) + 8:int(H * 0.72) + 38, 25:75] = 3  # 车 (略宽松)

# ---- 图 B: 多车场景 ----
img_b = Image.new('RGB', (W, H))
draw_b = ImageDraw.Draw(img_b)
draw_street_scene(draw_b, has_second_car=True, seed=42)

gt_b = np.zeros((H, W), dtype=np.int64)
gt_b[:int(H * 0.38)] = 0
gt_b[int(H * 0.38):int(H * 0.72)] = 1
gt_b[int(H * 0.72):] = 2
gt_b[int(H * 0.72) + 10:int(H * 0.72) + 35, 30:70] = 3
gt_b[int(H * 0.72) + 18:int(H * 0.72) + 43, 90:130] = 3
gt_b[int(H * 0.72) + 5:int(H * 0.72) + 30, 170:210] = 3

images = [('scene_a', img_a, gt_a), ('scene_b', img_b, gt_b)]

for name, img, gt in images:
    img.save(f'outputs/exp_{name}_input.png')
    print(f'  📁 已生成: outputs/exp_{name}_input.png ({img.size})')

# ============================================================
# 2. 初始化生成器 + 解码器
# ============================================================
gen = QwenGenerator(
    api_endpoint='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=os.environ['DASHSCOPE_API_KEY'],
    model_version='qwen-vl-max',
    api_mode='chat', timeout_sec=120, image_size=(W, H),
)
gen.colormap = CMAP
decoder = ColormapDecoder(colormap=CMAP, num_classes=4)

PROMPT = 'Analyze this street scene. Classes: sky(0), building(1), road(2), car(3). Output 16x16 grid JSON. Only JSON, no explanation.'

# ============================================================
# 3. 运行实验
# ============================================================
results = {}

for img_name, img_arr, gt in images:
    print(f'\n{"="*50}')
    print(f'📷 {img_name}')
    print(f'{"="*50}')

    if isinstance(img_arr, np.ndarray):
        img = Image.fromarray(img_arr)
    else:
        img = img_arr
    all_masks = []
    all_times = []

    for i in range(K):
        t0 = time.time()
        m = gen.generate(img, PROMPT)
        elapsed = time.time() - t0
        all_times.append(elapsed)
        cm = decoder.decode(np.array(m.convert('RGB')))
        all_masks.append(cm)
        print(f'  [{i+1}/{K}] {elapsed:.1f}s | 天空={np.sum(cm==0)} 建筑={np.sum(cm==1)} 道路={np.sum(cm==2)} 车辆={np.sum(cm==3)}')
        if i < K-1:
            time.sleep(2)  # 避免限流

    stack = np.stack(all_masks)

    # ---- PCU ----
    unc = build_uncertainty_map(stack, num_classes=4)
    print(f'  PCU: mean={unc.mean():.4f} max={unc.max():.4f} 高不确定像素={np.sum(unc>0.5)}')

    # ---- UGCF ----
    per_sample_unc = np.stack([build_uncertainty_map(stack[i:i+1], num_classes=4) for i in range(K)])
    fused = consensus_fusion(stack, per_sample_unc, method='ugcf', num_classes=4)
    fused_classes = [np.sum(fused==c) for c in range(4)]
    print(f'  UGCF:  天空={fused_classes[0]} 建筑={fused_classes[1]} 道路={fused_classes[2]} 车辆={fused_classes[3]}')

    # ---- Majority Vote ----
    from ugvu.consensus.majority_vote import majority_vote
    mv = majority_vote(stack)
    mv_classes = [np.sum(mv==c) for c in range(4)]
    print(f'  MVote:  天空={mv_classes[0]} 建筑={mv_classes[1]} 道路={mv_classes[2]} 车辆={mv_classes[3]}')

    # ---- Per-pixel agreement ----
    from ugvu.uncertainty.variance import agreement_ratio
    agree = agreement_ratio(stack)
    print(f'  Agreement: mean={agree.mean():.4f}')

    # ---- Calibration ----
    error_map = (mv != gt) & (gt != 255)
    ece = compute_ece_from_uncertainty(unc, error_map, num_bins=10)
    corr = correlation_report(unc, error_map)
    print(f'  ECE={ece["ece"]:.4f}  Spearman ρ={corr["spearman_r"]:.4f}  AUROC={corr["auroc"]:.4f}')

    # ---- mIoU vs GT ----
    from ugvu.metrics.metrics import mean_iou
    miou_mv = mean_iou(mv, gt, num_classes=4)
    miou_ugcf = mean_iou(fused, gt, num_classes=4)
    print(f'  mIoU(MV)={miou_mv:.4f}  mIoU(UGCF)={miou_ugcf:.4f}')

    results[img_name] = {
        'miou_mv': miou_mv, 'miou_ugcf': miou_ugcf,
        'ece': ece['ece'], 'spearman': corr['spearman_r'],
        'auroc': corr['auroc'], 'unc_mean': float(unc.mean()),
        'agree': float(agree.mean()), 'times': all_times,
        'fused_classes': fused_classes,
        'mv_classes': mv_classes,
    }

    # 保存可视化
    fused_vis = np.zeros((H, W, 3), dtype=np.uint8)
    for c in range(4):
        fused_vis[fused == c] = CMAP[c]
    Image.fromarray(fused_vis).save(f'outputs/exp_{img_name}_fused.png')
    Image.fromarray((unc * 255).astype(np.uint8)).save(f'outputs/exp_{img_name}_unc.png')
    Image.fromarray((error_map.astype(np.uint8) * 255)).save(f'outputs/exp_{img_name}_error.png')

# ============================================================
# 4. 综合报告
# ============================================================
print(f'\n{"="*60}')
print(f'  📊 综合实验报告')
print(f'{"="*60}')

for name, r in results.items():
    print(f'\n{name}:')
    print(f'  mIoU(MV)={r["miou_mv"]:.4f}  mIoU(UGCF)={r["miou_ugcf"]:.4f}')
    print(f'  ECE={r["ece"]:.4f}  Spearman ρ={r["spearman"]:.4f}  AUROC={r["auroc"]:.4f}')
    print(f'  Uncertainty mean={r["unc_mean"]:.4f}  Agreement={r["agree"]:.4f}')
    print(f'  API 耗时: 均值={np.mean(r["times"]):.1f}s ± {np.std(r["times"]):.1f}s')

# 保存 JSON 报告
# 转换 numpy 类型为 Python 原生类型
def to_native(obj):
    if isinstance(obj, dict): return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

with open('outputs/exp_report.json', 'w') as f:
    json.dump(to_native({k: {kk: vv for kk, vv in v.items() if kk != 'times'} for k, v in results.items()}), f, indent=2)

print(f'\n✅ 报告已保存: outputs/exp_report.json')
