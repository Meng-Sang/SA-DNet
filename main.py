import cv2

from utils.predict import sa_dnet

if __name__ == "__main__":
    ir_image = cv2.imread("assets/ir/5.jpg")
    vi_image = cv2.imread("assets/vi/5.jpg")
    match_image, fuse_image = sa_dnet(ir_image, vi_image, "person", is_mask=False)
    cv2.imwrite("assets/person_matching.jpg", match_image)
    cv2.imwrite("assets/person_unmask_fuse_image.jpg", fuse_image)
