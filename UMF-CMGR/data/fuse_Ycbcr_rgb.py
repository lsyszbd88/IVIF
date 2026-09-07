import os
import pathlib
import cv2
import numpy as np

if __name__ == '__main__':
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
    RESULTS_ROOT = PROJECT_ROOT / "results_Road"

    # 你的融合结果路径
    FUSED_DIR = str(RESULTS_ROOT / "Reg+Fusion/220508_cotrain_reg_fus/fused")
    VI_DIR    = str(DATASET_ROOT / "raw/ctest/Road/vi")
    RGB_OUT   = str(RESULTS_ROOT / "Reg+Fusion/220508_cotrain_reg_fus/fuse_rgb")

    if not os.path.exists(RGB_OUT):
        os.makedirs(RGB_OUT)

    y_file_list  = sorted(os.listdir(FUSED_DIR))
    vi_file_list = sorted(os.listdir(VI_DIR))

    for idx, (y_name, vi_name) in enumerate(zip(y_file_list, vi_file_list)):

        fuse_y_path = os.path.join(FUSED_DIR, y_name)
        vi_path     = os.path.join(VI_DIR, vi_name)

        # 读取融合灰度图 & 可见光彩色图
        fuse_y = cv2.imread(fuse_y_path, cv2.IMREAD_GRAYSCALE)
        img_vi = cv2.imread(vi_path, cv2.IMREAD_COLOR)

        # 如果尺寸不一致，resize 融合图到 VI 图大小
        if fuse_y.shape != img_vi.shape[:2]:
            fuse_y = cv2.resize(fuse_y, (img_vi.shape[1], img_vi.shape[0]))

        # VI → YCrCb，取 Cb/Cr，替换 Y 为融合结果
        vi_ycbcr = cv2.cvtColor(img_vi, cv2.COLOR_BGR2YCrCb)
        vi_cb = vi_ycbcr[:, :, 1]
        vi_cr = vi_ycbcr[:, :, 2]

        fused_ycbcr = np.stack([fuse_y, vi_cb, vi_cr], axis=2).astype(np.uint8)
        fused_bgr = cv2.cvtColor(fused_ycbcr, cv2.COLOR_YCrCb2BGR)

        cv2.imwrite(os.path.join(RGB_OUT, y_name), fused_bgr)

    print(f"完成！生成 {len(y_file_list)} 张彩色融合图 → {RGB_OUT}")
