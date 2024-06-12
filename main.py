import cv2
import torch

from model.fuse import get_ufusion
from model.reg import get_matchformer

if __name__ == "__main__":
    reg = get_matchformer()
    fuse = get_ufusion()
    ir_image = cv2.imread("assets/+1.jpg", cv2.IMREAD_GRAYSCALE)
    vi_image = cv2.imread("assets/-1.jpg", cv2.IMREAD_GRAYSCALE)
    vi_image = cv2.resize(vi_image, (640, 480))
    ir_image = cv2.resize(ir_image, (640, 480))
    with torch.no_grad():
        mkpts0, mkpts1 = reg(vi_image, ir_image)
        h, prediction = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 3.0, confidence=0.99999, maxIters=100000)
        d_ir_image = cv2.warpPerspective(ir_image, h, (640, 480))
        img = fuse(vi_image, d_ir_image)
        cv2.imshow("demo", img)
        cv2.waitKey(0)
