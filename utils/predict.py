import cv2
import numpy as np
import torch

from model.fuse import LazyUFusion
from model.reg import LazyMatchFormer
from model.sam import LazySAM
from utils.plot import make_matching_plot
from utils.utils import filter_points, letterbox_image, TPS


def sa_dnet(ir_image, vi_image, text, reg=LazyMatchFormer(), fuse=LazyUFusion(), sam=LazySAM(), size=None, is_sa=True,
            is_mask=False, is_fig_match=True):
    if size is None:
        size = (vi_image.shape[1], vi_image.shape[0])
    resize_ir_image = letterbox_image(ir_image, size)
    resize_vi_image = letterbox_image(vi_image, size)
    resize_ir_Y, resize_ir_Cr, resize_ir_Cb = cv2.split(cv2.cvtColor(resize_ir_image, cv2.COLOR_BGR2YCrCb))
    resize_vi_Y, resize_vi_Cr, resize_vi_Cb = cv2.split(cv2.cvtColor(resize_vi_image, cv2.COLOR_BGR2YCrCb))
    with torch.no_grad():
        ir_mkpst, vi_mkpst = reg(resize_ir_Y, resize_vi_Y)
        if is_sa:
            ir_mask = sam.predict(resize_ir_image, text)
            vi_mask = sam.predict(resize_vi_image, text)
            ir_mkpst, vi_mkpst = filter_points(ir_mask.sum(dim=0).numpy(), vi_mask.sum(dim=0).numpy(), ir_mkpst,
                                               vi_mkpst)
            if is_mask:
                resize_ir_Y = resize_ir_Y * (np.where(ir_mask.sum(dim=0) > 0, 1, 0).astype(np.uint8))
        d_ir_image = TPS(resize_ir_Y, ir_mkpst, vi_mkpst)
        fuse_image = fuse(d_ir_image, resize_vi_Y)
    match_image = None
    if is_fig_match:
        match_image = make_matching_plot(
            resize_ir_image, resize_vi_image, ir_mkpst, vi_mkpst, ir_mkpst, vi_mkpst,
            ["#86FC14" for i in range(ir_mkpst.shape[0])], "", path="assets/tmp.jpg")
    fuse_image = cv2.merge((fuse_image, resize_vi_Cr, resize_vi_Cb))
    bgr_fuse_image = cv2.cvtColor(fuse_image, cv2.COLOR_YCrCb2BGR)
    return match_image, bgr_fuse_image
