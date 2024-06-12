import torch

from .MatchFormer.config.defaultmf import default_cfg
from .MatchFormer.matchformer import MatchFormer


def get_matchformer(config=default_cfg):
    matchformer = MatchFormer(config)
    matchformer.load_state_dict(torch.load(r"model/reg/MatchFormer/weights/model.ckpt"),
                                strict=False)
    matchformer = matchformer.eval().cuda()
    return matchformer
