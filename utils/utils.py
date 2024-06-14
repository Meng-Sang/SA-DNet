import cv2
import numpy as np


def TPS(image, mkpts0, mkpts1):
    _, prediction = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 3, maxIters=100000)
    prediction = np.array(prediction, dtype=bool).reshape([-1])
    mkpts0 = mkpts0[prediction]
    mkpts1 = mkpts1[prediction]
    # 图像变化
    tps = cv2.createThinPlateSplineShapeTransformer()
    kp0 = mkpts0.reshape(1, -1, 2)
    kp1 = mkpts1.reshape(1, -1, 2)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(mkpts0))]
    tps.estimateTransformation(kp1, kp0, matches)
    print(kp1.shape)
    return tps.warpImage(image)


def filter_points(mask_ir, mask_vi, mkpts0, mkpts1):
    f_mask_ir = mask_ir.reshape(-1)
    f_mask_vi = mask_vi.reshape(-1)
    int_mkpts0 = mkpts0.astype(np.int32)
    int_mkpts1 = mkpts1.astype(np.int32)
    f_int_mkpts0 = int_mkpts0[:, 1] * mask_ir.shape[-1] + int_mkpts0[:, 0]
    f_int_mkpts1 = int_mkpts1[:, 1] * mask_vi.shape[-1] + int_mkpts1[:, 0]
    ir_sROI = f_mask_ir[f_int_mkpts0].astype(np.bool_)
    vi_sROI = f_mask_vi[f_int_mkpts1].astype(np.bool_)
    sROI = ir_sROI * vi_sROI
    index = np.argwhere(sROI > 0).reshape(-1)
    return mkpts0[index], mkpts1[index]


def letterbox_image(image, target_size):
    src_height, src_width = image.shape[:2]
    target_width, target_height = target_size
    scale = min(target_width / src_width, target_height / src_height)
    new_width = int(src_width * scale)
    new_height = int(src_height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_height, target_width, 3), 128, dtype=np.uint8)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    return canvas
