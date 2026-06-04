#!/usr/bin/env python3
"""
可视化 KataGo 全部训练指标，每个子图标题解释其含义。

每张图最多 2 行 × 3 列 = 3 个指标。上行全量趋势，下行最近 nsamp 窗口局部放大。
所有图片默认输出到 metrics_train.json 所在目录。

用法:
  python visualize/visualize_training_dynamics.py models/b6c96-train/metrics_train.json
  python visualize/visualize_training_dynamics.py models/b6c96-train/metrics_train.json --last-nsamp 300000
"""

import argparse
import json
import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

warnings.filterwarnings("ignore", message=".*does not have a glyph.*")

# ---------- 中文字体 ----------
def _setup_chinese_font():
    candidates = [
        "Noto Sans CJK JP",     # 含 U+2212 减号字形，完整 CJK 覆盖
        "Noto Serif CJK JP",
        "Noto Sans CJK SC",
        "Noto Serif CJK SC",
        "AR PL UMing CN",
        "AR PL UKai CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "SimHei",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            matplotlib.rcParams["font.family"] = [name, "DejaVu Sans"]
            return name
    for f in fm.fontManager.ttflist:
        if "CJK" in f.name or "UMing" in f.name or "UKai" in f.name:
            matplotlib.rcParams["font.family"] = [f.name, "DejaVu Sans"]
            return f.name
    print("WARNING: No Chinese font found, labels may render as boxes")
    return None

_FONT_NAME = _setup_chinese_font()
if _FONT_NAME:
    print(f"使用字体: {_FONT_NAME}")

# ---------- 指标分组，每组最多 3 个，共 19 组 ----------
# 格式: ("分组标题", [(key, "中文解释"), ...])

METRIC_GROUPS = [
    ("总损失 & 主要策略损失 (Total Loss & Main Policy)", [
        ("loss",        "总损失 (所有 loss 加权求和)"),
        ("p0loss",      "策略损失 - 我方 (交叉熵，模仿 MCTS visit counts)"),
        ("p1loss",      "策略损失 - 对手"),
    ]),

    ("软策略 & 策略准确率 (Soft Policy & Accuracy)", [
        ("p0softloss",  "软策略损失 - 我方 (温度 0.25 平滑，防止 overconfident)"),
        ("p1softloss",  "软策略损失 - 对手"),
        ("pacc1",       "策略 Top-1 准确率 (预测最佳着法与 MCTS 一致的比例)"),
    ]),

    ("长期乐观策略 (Long-term Optimistic Policy)", [
        ("p0lopt",      "长期乐观策略损失 (最终意外赢棋时，回头强化被低估的着法)"),
        ("p0loptw",     "长期乐观策略平均权重 (被标记为乐观的样本比例)"),
        ("p0sopt",      "短期乐观策略损失 (短期价值意外变好时强化)"),
    ]),

    ("短期乐观策略 & 策略熵 (Short-term Optimistic Policy & Entropy)", [
        ("p0soptw",     "短期乐观策略平均权重"),
        ("ptentr",      "策略熵 (越低说明策略越确定/尖锐)"),
        ("ptsoftentr",  "软策略熵 (温度 0.25 下的熵值)"),
    ]),

    ("价值损失 (Value Loss)", [
        ("vloss",       "价值损失 (预测最终胜负的交叉熵)"),
        ("vsquare",     "价值预测均方误差 MSE (值越小越准)"),
        ("tdvloss1",    "TD 价值损失 t=1 (预测 1 步后的短期价值)"),
    ]),

    ("TD 价值 & 分数损失 (TD Value & Score Losses)", [
        ("tdvloss2",    "TD 价值损失 t=2 (预测中期的价值变化)"),
        ("tdvloss3",    "TD 价值损失 t=3 (预测更长期的价值变化)"),
        ("tdsloss",     "TD 分数损失 (预测短期目数差变化)"),
    ]),

    ("领先 & 目数预测 (Lead & Score Prediction)", [
        ("leadloss",    "领先预测损失 (当前谁领先，领先多少)"),
        ("smloss",      "最终目数均值预测损失 (预测最终目差)"),
        ("sbcdfloss",   "分数分布 CDF 损失 (累积分布函数拟合)"),
    ]),

    ("分数分布 & 标准差 (Score Distribution)", [
        ("sbpdfloss",   "分数分布 PDF 损失 (概率密度函数拟合)"),
        ("sdregloss",   "分数标准差正则损失 (防止方差估计过大)"),
        ("oloss",       "归属权损失 (每个交叉点最终属于黑还是白)"),
    ]),

    ("空间输出 (Spatial Outputs)", [
        ("sloss",       "得分损失 (每个交叉点对最终目数的贡献值)"),
        ("fploss",      "未来位置损失 (预测后续几步棋子会落在哪里)"),
        ("skloss",      "双活检测损失 (识别哪些棋是双活状态)"),
    ]),

    ("双活权重 & 校准 (Seki Weight & Calibration)", [
        ("sekiweightscale", "双活损失动态权重 (训练初期低，后期升高)"),
        ("vtimeloss",   "方差时间损失 (局面不确定性随时间如何变化)"),
        ("evstloss",    "短期价值误差校准 (模型对 TD 价值估计的置信度)"),
    ]),

    ("误差校准 & Q 值 (Error Calibration & Q-Values)", [
        ("esstloss",    "短期分数误差校准 (模型对 TD 分数估计的置信度)"),
        ("qwlloss",     "Q 值走法质量损失 - 胜率维度 (v16+ 才用，v15 恒为 0)"),
        ("qscloss",     "Q 值走法质量损失 - 分数维度 (v16+ 才用，v15 恒为 0)"),
    ]),

    ("梯度范数 (Gradient Norm)", [
        ("gnorm_batch", "梯度总范数 (裁剪前，所有参数梯度的 L2 范数)"),
        ("gnorm_cap_batch", "梯度裁剪阈值 (超过此值的梯度会被裁剪)"),
        ("exgnorm",     "超出阈值的梯度部分 (越大说明梯度裁剪越激进)"),
    ]),

    ("总步长 & input/normal 步长 (Step Norms)", [
        ("step_norm_batch", "参数总更新步长 L2 (optimizer.step 前后的参数差)"),
        ("step_norm_input_batch", "input 层步长 (输入卷积层的更新幅度)"),
        ("step_norm_normal_batch", "normal 层步长 (主干残差块的更新幅度)"),
    ]),

    ("各参数组步长 1 (Per-Group Step Norms 1)", [
        ("step_norm_normal_gamma_batch", "normal_gamma 层步长 (BN 的 γ 缩放因子)"),
        ("step_norm_noreg_batch", "noreg 层步长 (无正则化的偏置等参数)"),
        ("step_norm_output_batch", "output 层步长 (输出头权重)"),
    ]),

    ("各参数组步长 2 & 范数 (Per-Group Step Norms 2 & Norms)", [
        ("step_norm_output_noreg_batch", "output_noreg 层步长 (输出头偏置)"),
        ("norm_input_batch", "input 层参数范数 (用于自适应 WD 计算)"),
        ("norm_normal_batch", "normal 层参数范数"),
    ]),

    ("模型参数范数 (Model Parameter Norms)", [
        ("norm_normal_gamma_batch", "normal_gamma 层范数"),
        ("norm_noreg_batch", "noreg 层范数"),
        ("norm_output_batch", "output 层范数"),
    ]),

    ("模型参数范数 2 (Model Norms 2)", [
        ("norm_output_noreg_batch", "output_noreg 层范数"),
        ("pslr_batch",  "每样本学习率 (含 warmup 缩放，控制每次更新的步幅)"),
        ("wdnormal_batch", "normal 层权重衰减值 (正则化强度)"),
    ]),

    ("数据窗口 (Data Window)", [
        ("window_start_batch", "数据窗口起始行号 (当前 epoch 数据范围的起点)"),
        ("window_end_batch", "数据窗口结束行号 (当前可用数据的总行数)"),
        ("wsum",        "累计权重和 (用于 EMA 指数滑动平均的归一化)"),
    ]),

    ("训练速度 (Training Speed)", [
        ("time_since_last_print", "输出间隔 (两次指标输出之间的秒数，反映训练速度)"),
    ]),
]


def load_data(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_nsamp(value):
    if not np.isfinite(value):
        return "nan"
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _set_ylim_to_data(ax, values):
    """纵轴自动缩到数据范围，不从 0 开始"""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return
    ymin, ymax = finite.min(), finite.max()
    if ymax == ymin:
        margin = 1.0
    else:
        margin = (ymax - ymin) * 0.05
    ax.set_ylim(ymin - margin, ymax + margin)


def _maybe_log_scale(ax, values):
    """值域 > 100x 且全正 → log scale"""
    pos = values[(values > 0) & np.isfinite(values)]
    if len(pos) > 1 and pos.max() / max(pos.min(), 1e-9) > 100:
        ax.set_yscale("log")


def make_figure(all_records, nsamp_values, group_title, metric_list, nsamp_window, output_path):
    """
    2 行 × 3 列:
    - 第 1 列: 指标 A (上行=全量, 下行=最近 nsamp 窗口)
    - 第 2 列: 指标 B
    - 第 3 列: 指标 C
    """
    n = len(metric_list)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 9))
    fig.suptitle(group_title, fontsize=14, fontweight="bold")

    # 确保 axes 是 2D 数组
    if n == 1:
        axes = axes.reshape(2, 1)

    finite_nsamp = nsamp_values[np.isfinite(nsamp_values)]
    x_end = finite_nsamp[-1]
    x_start = x_end - nsamp_window
    window_mask = (
        np.isfinite(nsamp_values)
        & (nsamp_values >= x_start)
        & (nsamp_values <= x_end)
    )

    for col, (key, label) in enumerate(metric_list):
        values = np.array([_to_float(r.get(key, float("nan"))) for r in all_records], dtype=np.float64)

        # ---- 上行: 全量趋势 ----
        ax_top = axes[0, col]
        ax_top.plot(nsamp_values, values, color="#1f77b4", linewidth=0.8)
        ax_top.set_title(f"全量 — {key}", fontsize=10, fontweight="bold")
        ax_top.set_xlabel("nsamp")
        ax_top.set_ylabel(label, fontsize=8)
        ax_top.grid(True, alpha=0.25)
        _set_ylim_to_data(ax_top, values)
        _maybe_log_scale(ax_top, values)

        # ---- 下行: 最近 nsamp 窗口 ----
        ax_bot = axes[1, col]
        x_last = nsamp_values[window_mask]
        v_last = values[window_mask]
        ax_bot.plot(x_last, v_last, color="#d62728", linewidth=1.0)
        ax_bot.set_title(
            f"nsamp {_format_nsamp(x_start)}–{_format_nsamp(x_end)} — {key}",
            fontsize=10,
            fontweight="bold",
        )
        ax_bot.set_xlabel("nsamp")
        ax_bot.set_ylabel(label, fontsize=8)
        ax_bot.grid(True, alpha=0.25)
        ax_bot.set_xlim(x_start, x_end)
        _set_ylim_to_data(ax_bot, v_last)
        _maybe_log_scale(ax_bot, v_last)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.basename(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="可视化 KataGo 全部训练指标")
    parser.add_argument("metrics_file", help="metrics_train.json 路径")
    parser.add_argument("--last-nsamp", "--last-n", dest="last_nsamp", type=int, default=6400000,
                        help="下行最近 nsamp 窗口大小 (默认 6400000，即 x-6400000 到 x；--last-n 为兼容旧别名)")
    args = parser.parse_args()

    if not os.path.exists(args.metrics_file):
        print(f"错误: 文件不存在 — {args.metrics_file}")
        sys.exit(1)
    if args.last_nsamp <= 0:
        print("错误: --last-nsamp 必须为正整数")
        sys.exit(1)

    records = load_data(args.metrics_file)
    if not records:
        print(f"错误: 文件没有可用记录 — {args.metrics_file}")
        sys.exit(1)
    print(f"加载 {len(records)} 条记录, {len(records[0])} 个指标")

    nsamp_values = np.array([_to_float(r.get("nsamp", float("nan"))) for r in records], dtype=np.float64)
    if not np.isfinite(nsamp_values).any():
        print("错误: 数据中没有可用的 nsamp，无法作为横坐标")
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.metrics_file))

    for i, (group_title, metric_list) in enumerate(METRIC_GROUPS):
        available = [(k, v) for k, v in metric_list if k in records[0]]
        if not available:
            print(f"跳过 (无数据): {group_title}")
            continue

        fname = f"training_dynamics_{i+1:02d}.png"
        out_path = os.path.join(out_dir, fname)
        print(f"[{i+1}/{len(METRIC_GROUPS)}] {group_title}")
        make_figure(records, nsamp_values, group_title, available, args.last_nsamp, out_path)

    print(f"\n完成，输出: {out_dir}/")


if __name__ == "__main__":
    main()
