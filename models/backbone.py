# models/backbone.py  （完整文件）
# ------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor, is_main_process
from util.fourier_ring import extract_ring_feat          # ★ NEW ★
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are frozen.
    """
    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        eps = 1e-5
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + eps).rsqrt()
        bias  = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self,
                 backbone: nn.Module,
                 train_backbone: bool,
                 num_channels: int,
                 return_interm_layers: bool,
                 add_ring: bool = True,
                 n_rho_bins: int = 4,
                 r_ratio: float = 0.15):
        super().__init__()
        self.add_ring   = add_ring
        self.n_rho_bins = n_rho_bins
        self.r_ratio    = r_ratio

        # 冻结低层
        for name, param in backbone.named_parameters():
            if (not train_backbone
               or ('layer2' not in name and 'layer3' not in name and 'layer4' not in name)):
                param.requires_grad_(False)

        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1",
                             "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels                     # 原 ResNet 通道

        if self.add_ring:
            self.ring_channels = self.num_channels * self.n_rho_bins
            self.num_channels += self.ring_channels          # 更新供 DETR 使用

    # --------------------------------------------------------
    def _add_ring_feat(self, x: torch.Tensor):
        """
        x : (B,C,H,W)  ->  concat ring feat → (B,C+Cʀɪɴɢ,H,W)
        """
        ring = extract_ring_feat(
            x, r_ratio=self.r_ratio, n_rho_bins=self.n_rho_bins)   # (B,C*Nρ,1,1)
        # broadcast 到 H×W 以便与 CNN 特征同形
        ring = ring.expand(-1, -1, x.shape[2], x.shape[3])
        return torch.cat([x, ring], dim=1)

    # --------------------------------------------------------
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)        # OrderedDict
        out: Dict[str, NestedTensor] = {}
        for name, feats in xs.items():
            mask = tensor_list.mask
            assert mask is not None
            mask = F.interpolate(mask[None].float(),
                                 size=feats.shape[-2:]).to(torch.bool)[0]

            if self.add_ring and name == list(xs.keys())[-1]:
                # 只给最高分辨率 (layer4) 加 F‑RIPE
                feats = self._add_ring_feat(feats)

            out[name] = NestedTensor(feats, mask)
        return out


class Backbone(BackboneBase):

    def __init__(self,
                 name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 add_ring: bool = True):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone,
                         train_backbone,
                         num_channels,
                         return_interm_layers,
                         add_ring=add_ring)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)          # -> dict[str→NestedTensor]
        out: List[NestedTensor] = []
        pos: List[torch.Tensor] = []
        for _, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone     = args.backbone 
    return_interm      = args.masks
    backbone = Backbone(args.backbone,
                        train_backbone,
                        return_interm,
                        args.dilation,
                        add_ring=True)          
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels  
    return model
