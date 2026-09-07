#!/usr/bin/env python3
"""
一键配置 RoadScene 数据集，使其匹配 UMF-CMGR 项目的目录结构。

用法:
    python setup_roadscene.py                          # 默认配置
    python setup_roadscene.py --ratio 0.45             # 自定义训练集比例
    python setup_roadscene.py --clone-dir /tmp/road    # 指定克隆位置

功能:
    1. 从 GitHub 克隆 RoadScene 仓库
    2. 按 train/test 划分图像 (默认 45%/55%)
    3. 创建软链接匹配项目期望的目录结构
"""

import argparse
import os
import pathlib
import random
import shutil
import subprocess
import sys


# ---------------------------------------------------------------------------
# 项目期望的目录结构（dataset 放在 UMF-CMGR 的同级目录 IVIF/ 下）
# ---------------------------------------------------------------------------
# dataset 根目录: UMF-CMGR/../ 即 IVIF/dataset/
DATASET_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent / "dataset"

EXPECTED_STRUCTURE = {
    "train": {
        "ir": "raw/ctrain/Road/ir",
        "vi": "raw/ctrain/Road/vi",
    },
    "test": {
        "ir": "raw/ctest/Road/ir",
        "vi": "raw/ctest/Road/vi",
    },
    # 以下目录由后续脚本生成，这里只创建空目录占位
    "generated": [
        "raw/ctrain/Road/ir_reg",
        "raw/ctrain/Road/ir_map",
        "raw/ctrain/Road/vi_map",
        "raw/ctrain/Road/it_edge",
        "raw/ctest/Road/ir_w",
        "raw/ctest/Road/it_edge",
        "raw/ctest/Road/disp",
        "raw/ctest/Road/ir_121",
        "test",
    ],
}

# RoadScene 仓库中 IR 和 VI 图像所在的子目录
# cropinfrared/ 和 crop_LR_visible/ 分辨率一致 (500×329)，且已空间对齐
REPO_IR_DIR = "cropinfrared"
REPO_VI_DIR = "crop_LR_visible"

RANDOM_SEED = 42


def get_project_root() -> pathlib.Path:
    """返回 UMF-CMGR/ 根目录"""
    return pathlib.Path(__file__).resolve().parent.parent


def clone_roadscene(clone_dir: pathlib.Path) -> pathlib.Path:
    """克隆 RoadScene 仓库，返回仓库路径"""
    repo_path = clone_dir / "RoadScene"
    if repo_path.exists():
        print(f"[跳过] RoadScene 已存在于 {repo_path}")
        return repo_path

    print(f"正在克隆 RoadScene 到 {clone_dir} ...")
    url = "https://github.com/hanna-xu/RoadScene.git"
    subprocess.run(
        ["git", "clone", url, str(repo_path)],
        check=True,
    )
    print("克隆完成。")
    return repo_path


def get_image_pairs(repo_path: pathlib.Path) -> dict:
    """获取 IR-VI 图像对，按 stem（不含扩展名的文件名）匹配。
    返回: {stem: {"ir": ir_filename, "vi": vi_filename}, ...}
    注：IR 是 .png，VI 是 .jpg，扩展名不同但 stem 相同。
    """
    ir_dir = repo_path / REPO_IR_DIR
    vi_dir = repo_path / REPO_VI_DIR

    VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

    # 构建 stem -> 文件名 的映射
    ir_map = {}
    for f in ir_dir.glob("*"):
        if f.suffix.lower() in VALID_SUFFIXES:
            ir_map[f.stem] = f.name

    vi_map = {}
    for f in vi_dir.glob("*"):
        if f.suffix.lower() in VALID_SUFFIXES:
            vi_map[f.stem] = f.name

    # 取 stem 交集，确保 IR 和 VI 一一对应
    common_stems = sorted(set(ir_map.keys()) & set(vi_map.keys()))

    pairs = {stem: {"ir": ir_map[stem], "vi": vi_map[stem]} for stem in common_stems}

    # 报告未匹配的
    ir_only = set(ir_map.keys()) - set(vi_map.keys())
    vi_only = set(vi_map.keys()) - set(ir_map.keys())
    if ir_only:
        print(f"[警告] 仅存在于 IR 的图像: {len(ir_only)} 张, 如: {list(ir_only)[:3]}")
    if vi_only:
        print(f"[警告] 仅存在于 VI 的图像: {len(vi_only)} 张, 如: {list(vi_only)[:3]}")

    print(f"找到 {len(pairs)} 对匹配的 IR-VI 图像")
    return pairs


def split_train_test(pairs: dict, train_ratio: float) -> tuple:
    """随机划分训练集和测试集，返回两个 dict: {stem: {ir, vi}, ...}"""
    random.seed(RANDOM_SEED)
    stems = list(pairs.keys())
    random.shuffle(stems)

    n_train = int(len(stems) * train_ratio)
    train_stems = sorted(stems[:n_train])
    test_stems = sorted(stems[n_train:])

    train_pairs = {s: pairs[s] for s in train_stems}
    test_pairs = {s: pairs[s] for s in test_stems}

    print(f"训练集: {len(train_pairs)} 张, 测试集: {len(test_pairs)} 张")
    return train_pairs, test_pairs


def copy_files(
    repo_path: pathlib.Path,
    pairs: dict,
    split_name: str,
):
    """直接复制图像文件到 IVIF/dataset/ 目录
    pairs: {stem: {"ir": ir_filename, "vi": vi_filename}, ...}
    """
    cfg = EXPECTED_STRUCTURE["train" if split_name == "train" else "test"]
    ir_dst = DATASET_ROOT / cfg["ir"]
    vi_dst = DATASET_ROOT / cfg["vi"]

    ir_src = repo_path / REPO_IR_DIR
    vi_src = repo_path / REPO_VI_DIR

    # 清空旧数据
    for d in [ir_dst, vi_dst]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for stem, info in pairs.items():
        ir_fname = info["ir"]  # e.g. FLIR_00006.jpg
        vi_fname = info["vi"]  # e.g. FLIR_00006.jpg

        shutil.copy2(ir_src / ir_fname, ir_dst / ir_fname)
        shutil.copy2(vi_src / vi_fname, vi_dst / vi_fname)

    print(f"  [{split_name}] 已复制 {len(pairs)} 对图像")


def create_generated_dirs():
    """创建后续脚本会生成的输出目录"""
    for rel_path in EXPECTED_STRUCTURE["generated"]:
        d = DATASET_ROOT / rel_path
        d.mkdir(parents=True, exist_ok=True)
    print("已创建所有输出目录占位。")


def print_summary():
    """打印最终的目录树概览"""
    print("\n" + "=" * 60)
    print("数据集目录结构概览:")
    print("=" * 60)
    if not DATASET_ROOT.exists():
        return

    for root, dirs, files in os.walk(DATASET_ROOT):
        level = root.replace(str(DATASET_ROOT), "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 3:
            sub_indent = "  " * (level + 1)
            for f in sorted(files)[:3]:
                print(f"{sub_indent}{f}")
            if len(files) > 3:
                print(f"{sub_indent}... 共 {len(files)} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description="配置 RoadScene 数据集以匹配 UMF-CMGR 项目结构"
    )
    parser.add_argument(
        "--clone-dir",
        type=pathlib.Path,
        default=None,
        help="RoadScene 仓库克隆位置 (默认: UMF-CMGR/../ 目录下)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.45,
        help="训练集比例 (默认 0.45，即 45%% 训练, 55%% 测试)",
    )
    args = parser.parse_args()

    project_root = get_project_root()
    print(f"项目根目录: {project_root}")

    clone_dir = args.clone_dir or (project_root.parent)
    clone_dir.mkdir(parents=True, exist_ok=True)

    # 1. 克隆 RoadScene
    repo_path = clone_roadscene(clone_dir)

    # 2. 获取图像对
    all_pairs = get_image_pairs(repo_path)

    # 3. 划分 train/test
    train_pairs, test_pairs = split_train_test(all_pairs, args.ratio)

    # 4. 复制图像文件
    print(f"\n数据将保存到: {DATASET_ROOT}")
    print("复制图像文件...")
    copy_files(repo_path, train_pairs, "train")
    copy_files(repo_path, test_pairs, "test")

    # 5. 创建后续需要的空目录
    create_generated_dirs()

    # 6. 打印结果
    print_summary()

    print("\n" + "=" * 60)
    print("配置完成！接下来可以按顺序运行:")
    print("  1. python data/get_svs_map.py          # 生成显著性图")
    print("  2. python data/get_test_data.py        # 生成测试用形变数据")
    print("  3. python Trainer/train_reg.py         # 训练配准网络")
    print("  4. python Trainer/train_fuse.py        # 训练融合网络")
    print("=" * 60)


if __name__ == "__main__":
    main()
