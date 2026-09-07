#!/usr/bin/env python3
"""Compute UMFusion baseline metrics on already saved RoadScene test results.

Registration (vs aligned IR GT `ir_121`): RMSE, NCC, LNCC
Fusion (fused vs IR/VI): EN, SD, SF, AG, MI, SSIM, CC, SCD, Qabf
"""
from __future__ import annotations

import csv
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT.parent / "dataset"
RESULTS = ROOT / "results_Road"
OUT_DIR = ROOT / "Evaluation"


def list_images(folder: Path) -> dict[str, Path]:
    files = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            files[p.stem] = p
    return files


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float64)


def matched_triplets(fused_dir: Path, ir_dir: Path, vi_dir: Path):
    fused = list_images(fused_dir)
    ir = list_images(ir_dir)
    vi = list_images(vi_dir)
    keys = sorted(set(fused) & set(ir) & set(vi))
    if not keys:
        raise RuntimeError(f"No matched names among {fused_dir}, {ir_dir}, {vi_dir}")
    return keys, fused, ir, vi


def entropy(img: np.ndarray) -> float:
    hist, _ = np.histogram(img.astype(np.uint8), bins=256, range=(0, 256))
    p = hist.astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def spatial_frequency(img: np.ndarray) -> float:
    rf = np.diff(img, axis=1)
    cf = np.diff(img, axis=0)
    r = np.sqrt((rf ** 2).mean())
    c = np.sqrt((cf ** 2).mean())
    return float(np.sqrt(r ** 2 + c ** 2))


def average_gradient(img: np.ndarray) -> float:
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt((gx ** 2 + gy ** 2) / 2.0)))


def corr_coeff(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    ha, _ = np.histogram(a.astype(np.uint8), bins=256, range=(0, 256), density=True)
    hb, _ = np.histogram(b.astype(np.uint8), bins=256, range=(0, 256), density=True)
    hab, _, _ = np.histogram2d(
        a.ravel(), b.ravel(), bins=256, range=[[0, 256], [0, 256]], density=True
    )
    ha = ha + 1e-12
    hb = hb + 1e-12
    hab = hab + 1e-12
    return float((hab * np.log2(hab / (ha[:, None] * hb[None, :]))).sum())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / 255.0
    b = b / 255.0
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu1 = cv2.filter2D(a, -1, window)
    mu2 = cv2.filter2D(b, -1, window)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(a ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(b ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(a * b, -1, window) - mu1_mu2
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float((num / (den + 1e-12)).mean())


def scd(fused: np.ndarray, ir: np.ndarray, vi: np.ndarray) -> float:
    return corr_coeff(fused - vi, ir) + corr_coeff(fused - ir, vi)


def qabf(ir: np.ndarray, vi: np.ndarray, fused: np.ndarray) -> float:
    """Xydeas & Petrovic Q^{AB/F}."""

    def sobel(img):
        sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        g = np.sqrt(sx ** 2 + sy ** 2)
        a = np.arctan2(sy, sx)
        return g, a

    def strength_orient(g1, a1, g2, a2):
        g = np.minimum(g1, g2) / (np.maximum(g1, g2) + 1e-12)
        a = np.abs(np.abs(a1 - a2) % math.pi - math.pi / 2) * 2 / math.pi
        qg = 0.9994 / (1 + np.exp(-15 * (g - 0.5)))
        qa = 0.9879 / (1 + np.exp(-22 * (a - 0.8)))
        return qg * qa

    ga, aa = sobel(ir)
    gb, ab = sobel(vi)
    gf, af = sobel(fused)
    qaf = strength_orient(ga, aa, gf, af)
    qbf = strength_orient(gb, ab, gf, af)
    wa = ga ** 1.0
    wb = gb ** 1.0
    return float(((qaf * wa + qbf * wb).sum()) / (wa + wb + 1e-12).sum())


def ncc_global(a: np.ndarray, b: np.ndarray) -> float:
    return corr_coeff(a, b)


def lncc(a: np.ndarray, b: np.ndarray, win: int = 17) -> float:
    ia = torch.from_numpy(a).float()[None, None] / 255.0
    ib = torch.from_numpy(b).float()[None, None] / 255.0
    filt = torch.ones(1, 1, win, win)
    pad = win // 2
    stride = 1
    i_sum = F.conv2d(ia, filt, stride=stride, padding=pad)
    j_sum = F.conv2d(ib, filt, stride=stride, padding=pad)
    i2_sum = F.conv2d(ia * ia, filt, stride=stride, padding=pad)
    j2_sum = F.conv2d(ib * ib, filt, stride=stride, padding=pad)
    ij_sum = F.conv2d(ia * ib, filt, stride=stride, padding=pad)
    win_size = float(win * win)
    u_i = i_sum / win_size
    u_j = j_sum / win_size
    cross = ij_sum - u_j * i_sum - u_i * j_sum + u_i * u_j * win_size
    i_var = i2_sum - 2 * u_i * i_sum + u_i * u_i * win_size
    j_var = j2_sum - 2 * u_j * j_sum + u_j * u_j * win_size
    cc = cross * cross / (i_var * j_var + 1e-5)
    return float(cc.mean())


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def eval_registration(pred_dir: Path, gt_dir: Path, name: str) -> dict:
    pred = list_images(pred_dir)
    gt = list_images(gt_dir)
    keys = sorted(set(pred) & set(gt))
    rows = []
    for k in keys:
        p = read_gray(pred[k])
        g = read_gray(gt[k])
        if p.shape != g.shape:
            g = cv2.resize(g, (p.shape[1], p.shape[0]), interpolation=cv2.INTER_AREA)
        mse = float(np.mean((p - g) ** 2) / (255.0 ** 2))
        rmse = math.sqrt(mse)
        rows.append(
            {
                "name": k,
                "MSE": mse,
                "RMSE": rmse,
                "NCC": ncc_global(p, g),
                "LNCC": lncc(p, g),
            }
        )
    summary = {"task": "registration", "split": name, "n": len(rows)}
    for key in ["MSE", "RMSE", "NCC", "LNCC"]:
        m, s = mean_std([r[key] for r in rows])
        summary[key] = m
        summary[f"{key}_std"] = s
    return summary, rows


def eval_fusion(fused_dir: Path, ir_dir: Path, vi_dir: Path, name: str) -> dict:
    keys, fused, ir, vi = matched_triplets(fused_dir, ir_dir, vi_dir)
    rows = []
    for k in keys:
        f = read_gray(fused[k])
        a = read_gray(ir[k])
        b = read_gray(vi[k])
        if a.shape != f.shape:
            a = cv2.resize(a, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_AREA)
        if b.shape != f.shape:
            b = cv2.resize(b, (f.shape[1], f.shape[0]), interpolation=cv2.INTER_AREA)
        rows.append(
            {
                "name": k,
                "EN": entropy(f),
                "SD": float(f.std()),
                "SF": spatial_frequency(f),
                "AG": average_gradient(f),
                "MI": mutual_information(f, a) + mutual_information(f, b),
                "SSIM": 0.5 * (ssim(f, a) + ssim(f, b)),
                "CC": 0.5 * (corr_coeff(f, a) + corr_coeff(f, b)),
                "SCD": scd(f, a, b),
                "Qabf": qabf(a, b, f),
            }
        )
    summary = {"task": "fusion", "split": name, "n": len(rows)}
    for key in ["EN", "SD", "SF", "AG", "MI", "SSIM", "CC", "SCD", "Qabf"]:
        m, s = mean_std([r[key] for r in rows])
        summary[key] = m
        summary[f"{key}_std"] = s
    return summary, rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(v: float) -> str:
    return f"{v:.4f}"


def main():
    jobs = [
        (
            "reg_only",
            RESULTS / "Reg/220507_Deformable_2*Fe_10*Grad/ir_reg",
            DATASET / "raw/ctest/Road/ir_121",
        ),
        (
            "reg_joint",
            RESULTS / "Reg+Fusion/220508_cotrain_reg_fus/ir_reg",
            DATASET / "raw/ctest/Road/ir_121",
        ),
    ]
    fuse_jobs = [
        (
            "fusion_only",
            RESULTS / "Fusion/220507_fusion_w_svm/fused",
            RESULTS / "Fusion/220507_fusion_w_svm/ir",
            RESULTS / "Fusion/220507_fusion_w_svm/vi",
        ),
        (
            "fusion_joint",
            RESULTS / "Reg+Fusion/220508_cotrain_reg_fus/fused",
            RESULTS / "Reg+Fusion/220508_cotrain_reg_fus/ir_reg",
            RESULTS / "Reg+Fusion/220508_cotrain_reg_fus/vi",
        ),
    ]

    summaries = []
    lines = ["# UMFusion RoadScene baseline metrics", "", f"Test images are matched by filename stem.", ""]

    lines.append("## Registration (pred IR vs aligned GT `ir_121`)")
    lines.append("")
    lines.append("| Setting | N | RMSE ↓ | NCC ↑ | LNCC ↑ | MSE ↓ |")
    lines.append("|---------|---|--------|-------|--------|-------|")
    for name, pred, gt in jobs:
        print(f"[reg] {name} ...", flush=True)
        summary, rows = eval_registration(pred, gt, name)
        summaries.append(summary)
        write_csv(OUT_DIR / f"reg_{name}.csv", rows)
        lines.append(
            f"| {name} | {summary['n']} | {fmt(summary['RMSE'])} | {fmt(summary['NCC'])} | {fmt(summary['LNCC'])} | {fmt(summary['MSE'])} |"
        )

    lines += ["", "## Fusion (fused vs IR/VI)", ""]
    lines.append("| Setting | N | EN ↑ | SD ↑ | SF ↑ | AG ↑ | MI ↑ | SSIM ↑ | CC ↑ | SCD ↑ | Qabf ↑ |")
    lines.append("|---------|---|------|------|------|------|------|--------|------|-------|--------|")
    for name, fused, ir, vi in fuse_jobs:
        print(f"[fuse] {name} ...", flush=True)
        summary, rows = eval_fusion(fused, ir, vi, name)
        summaries.append(summary)
        write_csv(OUT_DIR / f"fuse_{name}.csv", rows)
        lines.append(
            "| {split} | {n} | {EN} | {SD} | {SF} | {AG} | {MI} | {SSIM} | {CC} | {SCD} | {Qabf} |".format(
                split=name,
                n=summary["n"],
                **{k: fmt(summary[k]) for k in ["EN", "SD", "SF", "AG", "MI", "SSIM", "CC", "SCD", "Qabf"]},
            )
        )

    lines += [
        "",
        "Notes:",
        "- `reg_only`: DeformableNet trained 800 epochs, tested on warped IR (`ir_w`).",
        "- `fusion_only`: FusionNet trained 200 epochs on registered IR (`ir_reg`).",
        "- `*_joint`: jointly trained 600 epochs after loading the two pretrained nets.",
        "- `fusion_joint` uses registered IR (`ir_reg`) rather than the warped input IR.",
        "- Higher is better except RMSE/MSE.",
        "",
    ]
    out_md = OUT_DIR / "baseline_metrics.md"
    out_md.write_text("\n".join(lines) + "\n")
    write_csv(OUT_DIR / "summary.csv", summaries)
    print(out_md.read_text())


if __name__ == "__main__":
    sys.exit(main())
