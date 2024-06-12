import torch

from .UFusion.backbone.UFusion import UFusion


def get_ufusion():
    ufusion = UFusion()
    ufusion.load_state_dict(torch.load(f"model/fuse/UFusion/weights/model.pth"))
    ufusion = ufusion.eval().cuda()
    return ufusion
