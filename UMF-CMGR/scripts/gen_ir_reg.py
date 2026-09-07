#!/usr/bin/env python3
"""用训练好的 RegNet 对训练集推理，生成 ir_reg。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pathlib
import cv2, numpy as np, kornia, torch
from tqdm import tqdm
from models.deformable_net import DeformableNet

IR_DIR   = pathlib.Path("/home/liusy/IVIF/dataset/raw/ctrain/Road/ir")
IT_DIR   = pathlib.Path("/home/liusy/IVIF/dataset/raw/ctrain/Road/it_edge")
DST_DIR  = pathlib.Path("/home/liusy/IVIF/dataset/raw/ctrain/Road/ir_reg")
CKPT     = "/home/liusy/IVIF/UMF-CMGR/cache/Reg_only/220507_Deformable_2*Fe_10*Grad/cp_0800.pth"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DeformableNet().to(device)
    net.load_state_dict(torch.load(CKPT))
    net.eval()

    ir_files = sorted(IR_DIR.glob("*"))
    it_files = sorted(IT_DIR.glob("*"))
    DST_DIR.mkdir(parents=True, exist_ok=True)

    for ir_path, it_path in tqdm(zip(ir_files, it_files), total=len(ir_files)):
        ir = imread(ir_path).to(device)
        it = imread(it_path).to(device)

        _, _, h, w = ir.shape
        pad_h = (4 - h % 4) % 4
        pad_w = (4 - w % 4) % 4
        ir_pad = torch.nn.functional.pad(ir, (0, pad_w, 0, pad_h), mode='reflect') if pad_h or pad_w else ir
        it_pad = torch.nn.functional.pad(it, (0, pad_w, 0, pad_h), mode='reflect') if pad_h or pad_w else it

        with torch.no_grad():
            ir_pred, _, _, _, _, _ = net(it_pad, ir_pad, shape=(h + pad_h, w + pad_w))
            ir_pred = ir_pred[:, :, :h, :w]

        imsave(ir_pred, DST_DIR / ir_path.name)

    print(f"完成！生成 {len(ir_files)} 张 ir_reg → {DST_DIR}")

def imread(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return kornia.utils.image_to_tensor(im / 255.).type(torch.FloatTensor).unsqueeze(0)

def imsave(tensor, path):
    im = kornia.utils.tensor_to_image(tensor.squeeze().cpu()) * 255.
    cv2.imwrite(str(path), im)

if __name__ == "__main__":
    main()
