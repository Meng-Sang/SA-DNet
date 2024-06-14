import cv2

from utils.predict import sa_dnet

if __name__ == "__main__":
    ir_image = cv2.imread("assets/ir/006393.jpg")
    vi_image = cv2.imread("assets/vi/006393.jpg")
    _, fuse_image = sa_dnet(ir_image, vi_image, "car", is_mask=True)
    cv2.imwrite("demo.jpg", fuse_image)
