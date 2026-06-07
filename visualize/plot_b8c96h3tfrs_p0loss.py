#!/usr/bin/env python3

import argparse
import glob
import json
import math
import os
import warnings

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings("ignore", message=".*does not have a glyph.*")


def setup_font():
    candidates = [
        "Noto Sans CJK JP",
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
    return None


def read_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def label_for_path(path):
    name = os.path.basename(os.path.dirname(path))
    prefix = "attnrescmp-"
    if name.startswith(prefix):
        name = name[len(prefix) :]
    suffix = "-bs256-s5m-20260606"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    if "-s" in name and "-d" in name:
        name = name.split("-s", 1)[0]
    labels = {
        "b8c96h3tfrs": "base",
        "b8c96h3tfrs_cos": "base cosine",
        "b8c96h3tfrsattnres": "attnres",
        "b8c96h3tfrsattnres_cos": "attnres cosine",
    }
    return labels.get(name, name)


def finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def set_ylim_to_data(ax, arrays):
    values = np.concatenate([arr[np.isfinite(arr)] for arr in arrays if len(arr) > 0])
    if len(values) == 0:
        return
    ymin = values.min()
    ymax = values.max()
    margin = 1.0 if ymax == ymin else (ymax - ymin) * 0.05
    ax.set_ylim(ymin - margin, ymax + margin)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--skip-lines", type=int, default=150)
    parser.add_argument(
        "--pattern",
        default="models/b8c96h3tfrs*/metrics_train.json",
    )
    parser.add_argument(
        "--output",
        default="visualize/b8c96h3tfrs_p0loss_skip150.png",
    )
    args = parser.parse_args()

    setup_font()

    paths = sorted(glob.glob(os.path.join(args.root, args.pattern)))
    if not paths:
        raise SystemExit(f"No metrics files matched: {args.pattern}")

    series = []
    skipped = []
    for path in paths:
        records = read_records(path)
        visible = records[args.skip_lines :]
        label = label_for_path(path)
        if not visible:
            skipped.append((label, len(records), "no rows after skip"))
            continue

        xs = np.array([finite_float(r.get("nsamp")) / 1_000_000.0 for r in visible])
        ys = np.array([finite_float(r.get("p0loss")) for r in visible])
        mask = np.isfinite(xs) & np.isfinite(ys)
        if not mask.any():
            skipped.append((label, len(records), "missing nsamp or p0loss"))
            continue
        series.append((label, xs[mask], ys[mask], len(records)))

    if not series:
        raise SystemExit("No drawable p0loss series after skipping rows")

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for label, xs, ys, total_rows in series:
        marker_every = max(1, len(xs) // 10)
        ax.plot(
            xs,
            ys,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            markevery=marker_every,
            label=f"{label} ({len(xs)}/{total_rows} rows)",
        )
        ax.annotate(
            f"{ys[-1]:.4f}",
            xy=(xs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
        )

    set_ylim_to_data(ax, [ys for _, _, ys, _ in series])
    ax.set_title("b8c96h3tfrs p0loss comparison, first 150 metric rows skipped", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training samples / 1e6")
    ax.set_ylabel("p0loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    output = os.path.join(args.root, args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"saved: {output}")
    print("plotted:")
    for label, xs, ys, total_rows in series:
        print(f"  {label}: rows={total_rows}, plotted={len(xs)}, final_nsamp={xs[-1]:.6g}M, final_p0loss={ys[-1]:.6g}")
    if skipped:
        print("skipped:")
        for label, rows, reason in skipped:
            print(f"  {label}: rows={rows}, reason={reason}")


if __name__ == "__main__":
    main()
