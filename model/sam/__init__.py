import cv2
import numpy as np
import torch
from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops
from segment_anything import sam_model_registry
from segment_anything import SamPredictor
import groundingdino.datasets.transforms as T

from configs.config import DEVICE


def transform_image(image) -> torch.Tensor:
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image_transformed, _ = transform(image, None)
    return image_transformed


class LazySAM:

    def __init__(self):
        super(LazySAM, self).__init__()
        self.dino_model = None
        self.seg_any_model = None
        self.device = DEVICE

    @staticmethod
    def dino_pre_process(image):
        image = Image.fromarray(image)
        image_trans = transform_image(image)
        return image_trans

    def dino_predict(self, image, text, box_threshold=0.3, text_threshold=0.25):
        if self.dino_model is None:
            self.dino_model = load_model("model/sam/grounding_dino/config/GroundingDINO_SwinB_cfg.py",
                                         "model/sam/grounding_dino/weight/model.pth")
        image_trans = self.dino_pre_process(image)
        boxes, _, _ = predict(
            model=self.dino_model,
            image=image_trans,
            caption=text,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        H, W, _ = image.shape
        boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
        return boxes

    def seg_any_predict(self, image, boxes):
        if self.seg_any_model is None:
            self.seg_any_model = sam_model_registry["vit_h"](checkpoint=f"model/sam/segment_anything/weight/model.pth")
            self.seg_any_model.to(device=self.device)
            self.seg_any_model = SamPredictor(self.seg_any_model)
        image_array = np.asarray(image)
        self.seg_any_model.set_image(image_array)
        transformed_boxes = self.seg_any_model.transform.apply_boxes_torch(boxes, image_array.shape[:2])
        masks, _, _ = self.seg_any_model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(self.device),
            multimask_output=False,
        )
        return masks.cpu()

    def predict(self, image, text):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = self.dino_predict(image, text)
        H, W, _ = image.shape
        if len(boxes) < 1:
            boxes = torch.ones((1, 4)) * torch.Tensor([W, H, W, H])
        masks = self.seg_any_predict(image, boxes)
        masks = masks.squeeze(1)
        return masks.cpu()


if __name__ == "__main__":
    sam = LazySAM()
