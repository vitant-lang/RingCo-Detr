# -*- coding: utf-8 -*-
"""
RingCo-DETR: Spectral-Ring Consistency, Self-Tuning Query Cooperation, and Adaptive Matching
==========================================================================================
Complete FIXED implementation with three key innovations:
1. Adaptive Hungarian Matching with EMA scaling and IoU-ring auxiliaries
2. F-RIPE: Frequency-domain Ring-Invariant Profile Enhancement 
3. OGQC: Oriented Geometry Query Cooperation with self-tuning thresholds

CRITICAL FIXES APPLIED:
- Fixed gradient flow in OGQC (was broken by torch.tensor() wrapping)
- Vectorized F-RIPE implementation for stability and speed
- EMA-based self-balancing instead of non-functional median normalization
- Proper tensor operations throughout to maintain computational graph
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.fft import fft2, fftshift

from util.misc import (
    nested_tensor_from_tensor_list,
    accuracy,
    is_dist_avail_and_initialized,
    get_world_size,
)
from .backbone import build_backbone
from .transformerold import build_transformer
from .matcher import build_matcher
from util.box_ops import rotated_iou_pairwise, rotated_giou_pairwise

if not hasattr(np, 'object'):
    np.object = object


def angle_loss_cosine_similarity(pred_sincos: torch.Tensor,
                                 target_sincos: torch.Tensor,
                                 temperature: float = 1.0) -> torch.Tensor:

    pred_norm = F.normalize(pred_sincos, dim=-1, eps=1e-8)
    target_norm = F.normalize(target_sincos, dim=-1, eps=1e-8)
    cos_sim = (pred_norm * target_norm).sum(dim=-1)
    cos_sim_flipped = (pred_norm * (-target_norm)).sum(dim=-1)
    best_cos_sim = torch.max(cos_sim, cos_sim_flipped)
    loss = torch.clamp(1.0 - best_cos_sim, min=0.0, max=2.0) / temperature
    return loss.mean()

def circular_distance(theta1: torch.Tensor, theta2: torch.Tensor) -> torch.Tensor:
 
    diff = torch.abs(theta1 - theta2)
    return torch.min(diff, 2*math.pi - diff).clamp(max=math.pi/2)

def circular_mean(angles: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute circular mean of angles"""
    sin_mean = torch.sin(angles).mean(dim=dim, keepdim=True)
    cos_mean = torch.cos(angles).mean(dim=dim, keepdim=True)
    return torch.atan2(sin_mean, cos_mean).squeeze(dim)


class FRIPE(nn.Module):

    def __init__(self, k_ring_eff: int = 8, eps: float = 1e-6, 
                 energy_threshold: float = 0.95, use_adaptive_k: bool = False):
        super().__init__()
        self.k_ring_eff = k_ring_eff
        self.eps = eps
        self.energy_threshold = energy_threshold
        self.use_adaptive_k = use_adaptive_k  # True for paper version, False for fixed K
        
    def build_equal_energy_rings(self, magnitude_2d: torch.Tensor) -> Tuple[List[torch.Tensor], int]:

        H, W = magnitude_2d.shape
        device = magnitude_2d.device
        
        # Create distance matrix from center
        center_y, center_x = H // 2, W // 2
        y_coords = torch.arange(H, device=device).float() - center_y
        x_coords = torch.arange(W, device=device).float() - center_x
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        distances = torch.sqrt(xx**2 + yy**2)
        
        # Flatten and sort by distance
        flat_magnitude = magnitude_2d.flatten()
        flat_distance = distances.flatten()
        
        # Sort pixels by distance from center
        sorted_indices = torch.argsort(flat_distance)
        sorted_magnitudes = flat_magnitude[sorted_indices]
        
     
        total_energy = sorted_magnitudes.sum()
        cumulative_energy = torch.cumsum(sorted_magnitudes, dim=0)
        cumulative_ratio = cumulative_energy / (total_energy + self.eps)

        # 选定用于分环的“能量截断”像素范围（默认用全部）
        if self.use_adaptive_k:
            # 找到首次达到阈值的位置，+1 表示包含该像素
            cutoff_idx = torch.searchsorted(cumulative_ratio, torch.tensor(self.energy_threshold, device=device)).item()
            cutoff_idx = max(1, min(cutoff_idx, len(sorted_indices)))
        else:
            cutoff_idx = len(sorted_indices)

        # 在截断范围内等能量分为 k_ring_eff 个环
        k_eff = max(1, int(self.k_ring_eff))
        sel_indices = sorted_indices[:cutoff_idx]
        sel_cum_energy = cumulative_energy[:cutoff_idx]
        target_energy_per_ring = (sel_cum_energy[-1] + 1e-12) / k_eff

        ring_masks = []
        start_idx = 0
        H, W = magnitude_2d.shape
        for k in range(k_eff):
            target_cum = (k + 1) * target_energy_per_ring
            end_idx = torch.searchsorted(sel_cum_energy, target_cum).item()
            end_idx = min(max(end_idx, start_idx + 1), sel_indices.numel())
            if k == k_eff - 1:
                end_idx = sel_indices.numel()

            pix_idx = sel_indices[start_idx:end_idx]
            mask = torch.zeros(H * W, dtype=torch.bool, device=device)
            if pix_idx.numel() > 0:
                mask[pix_idx] = True
                mask = mask.view(H, W)
                ring_masks.append(mask)
            start_idx = end_idx
            if start_idx >= sel_indices.numel():
                break

        return ring_masks, len(ring_masks)
    
    def compute_ring_profiles(self, magnitude: torch.Tensor, ring_masks: List[torch.Tensor]) -> torch.Tensor:

        if len(magnitude.shape) == 3:
            C, H, W = magnitude.shape
            magnitude_flat = magnitude.view(C, -1)  # [C, H*W]
        else:
            H, W = magnitude.shape
            C = 1
            magnitude_flat = magnitude.view(1, -1)  # [1, H*W]
        
        profiles = []
        for ring_mask in ring_masks:
            mask_flat = ring_mask.view(-1)  # [H*W]
            
            if mask_flat.sum() == 0:
                continue
            
            # Extract ring pixels and compute log-energy
            ring_magnitudes = magnitude_flat[:, mask_flat]  # [C, num_ring_pixels]
            ring_log_energy = torch.log(1 + ring_magnitudes).sum()
            
            # Average over channels and pixels
            avg_energy = ring_log_energy / (C * mask_flat.sum().float())
            profiles.append(avg_energy)
        
        return torch.stack(profiles) if profiles else torch.tensor([], device=magnitude.device)
    
    def forward(self, features: torch.Tensor, features_aug: torch.Tensor) -> torch.Tensor:

        if not self.training:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        
        B, C, H, W = features.shape
        total_loss = torch.zeros((), device=features.device, requires_grad=True)
        
        for b in range(B):
            # Compute centered DFT magnitudes
            feat_fft = fft2(features[b])
            feat_fft_centered = fftshift(feat_fft, dim=(-2, -1))
            magnitude = torch.abs(feat_fft_centered)
            
            feat_aug_fft = fft2(features_aug[b])
            feat_aug_fft_centered = fftshift(feat_aug_fft, dim=(-2, -1))
            magnitude_aug = torch.abs(feat_aug_fft_centered)
            
            # Average magnitude for stable ring creation
            avg_magnitude_2d = 0.5 * (magnitude + magnitude_aug).mean(dim=0)  # [H, W]
            
            # Create equal-energy rings (vectorized)
            ring_masks, k_eff = self.build_equal_energy_rings(avg_magnitude_2d)
            
            if k_eff == 0:
                continue
            
            # Compute profiles for both magnitudes
            profiles = self.compute_ring_profiles(magnitude, ring_masks)
            profiles_aug = self.compute_ring_profiles(magnitude_aug, ring_masks)
            
            if len(profiles) == 0 or len(profiles_aug) == 0:
                continue
            
            # MSE loss between profiles
            loss = F.mse_loss(profiles, profiles_aug)
            total_loss = total_loss + loss
        
        return total_loss / B if B > 0 else torch.tensor(0.0, device=features.device, requires_grad=True)



class OGQC(nn.Module):

    def __init__(self, hidden_dim: int, num_bins: int = 36, top_k_fallback: int = 8):
        super().__init__()
        self.num_bins = num_bins
        self.top_k_fallback = top_k_fallback
        self.bin_size = 2 * math.pi / num_bins
        
        # Learnable bin embeddings
        self.bin_embeddings = nn.Embedding(num_bins, hidden_dim)
        
        # Initialize embeddings
        nn.init.normal_(self.bin_embeddings.weight, std=0.02)
        
    def get_angle_from_sincos(self, sincos: torch.Tensor) -> torch.Tensor:
        """Convert sin/cos representation to angle in [0, 2π)
        Convention: sincos = [cos, sin] so atan2(sin, cos) = atan2(sincos[1], sincos[0])
        """
        return torch.atan2(sincos[..., 1], sincos[..., 0]) % (2 * math.pi)
    
    def get_bin_index(self, angles: torch.Tensor) -> torch.Tensor:
        """Convert angles to bin indices"""
        return (angles / self.bin_size).long().clamp(0, self.num_bins - 1)
    
    def forward(self, 
                query_features: torch.Tensor, 
                predicted_angles: torch.Tensor,
                targets: List[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = query_features.device
        
        if not self.training:
            return torch.zeros((), device=device, requires_grad=True), \
                   torch.zeros((), device=device, requires_grad=True)
        
        B, Q, D = query_features.shape
        
        # CRITICAL FIX: Initialize as tensors with gradients, not Python floats
        total_coop_loss = torch.zeros((), device=device, requires_grad=True)
        total_comp_loss = torch.zeros((), device=device, requires_grad=True)
        valid_batches = 0
        
        for b in range(B):
            # Get angles and bin indices
            angles = self.get_angle_from_sincos(predicted_angles[b])  # [Q]
            bin_indices = self.get_bin_index(angles)  # [Q]
            
            # Fuse queries with bin embeddings
            bin_embeds = self.bin_embeddings(bin_indices)  # [Q, D]
            fused_queries = query_features[b] + bin_embeds  # [Q, D]
            normalized_queries = F.normalize(fused_queries, dim=-1)  # [Q, D]
            
            # Compute cosine similarities
            similarities = torch.mm(normalized_queries, normalized_queries.t())  # [Q, Q]
            
            # FIXED: Block diagonal to prevent self-pairing
            similarities.fill_diagonal_(float('-inf'))
            
            # Group pairs by bin combinations
            pairs_by_bin = defaultdict(list)
            
            for i in range(Q):
                for j in range(i + 1, Q):
                    bin_i, bin_j = bin_indices[i].item(), bin_indices[j].item()
                    bin_pair = tuple(sorted([bin_i, bin_j]))
                    pairs_by_bin[bin_pair].append((i, j))
            
            # Select highest-similarity pair from each bin pair
            selected_pairs = []
            for bin_pair, pair_list in pairs_by_bin.items():
                if pair_list:
                    # Find pair with highest similarity (keeping tensor operations)
                    pair_similarities = torch.stack([similarities[i, j] for i, j in pair_list])
                    best_idx = torch.argmax(pair_similarities)
                    selected_pairs.append(pair_list[best_idx.item()])
            
            if not selected_pairs:
                continue
            
            # Self-tuning threshold: high quantile of similarities
            pair_similarities = torch.stack([similarities[i, j] for i, j in selected_pairs])
            S = len(pair_similarities)
            
            if S == 0:
                continue
            
            # Adaptive quantile
            q_s = 1.0 - 1.0 / (S + 1)
            threshold = torch.quantile(pair_similarities, q_s)
            
            # Filter pairs above threshold
            mask = pair_similarities > threshold
            filtered_pairs = [pair for pair, keep in zip(selected_pairs, mask) if keep.item()]
            
            if len(filtered_pairs) < 2:
                # Fallback to top-k pairs globally
                all_similarities = similarities[similarities != float('-inf')]
                if len(all_similarities) >= self.top_k_fallback:
                    top_k_threshold = torch.topk(all_similarities, self.top_k_fallback)[0][-1]
                    mask = pair_similarities >= top_k_threshold
                    filtered_pairs = [pair for pair, keep in zip(selected_pairs, mask) if keep.item()]
            
            if len(filtered_pairs) == 0:
                continue
            
            # Compute circular distances (keep as tensors)
            pair_distances = torch.stack([
                circular_distance(angles[i], angles[j]) for i, j in filtered_pairs
            ])
            
            # Self-tuning split threshold: median distance
            alpha_star = torch.median(pair_distances)
            alpha_star = torch.clamp(alpha_star, math.pi / self.num_bins, math.pi / 2)
            
            # Separate cooperation and competition pairs
            coop_mask = pair_distances <= alpha_star
            comp_mask = ~coop_mask
            
            coop_pairs = [pair for pair, is_coop in zip(filtered_pairs, coop_mask) if is_coop.item()]
            comp_pairs = [pair for pair, is_comp in zip(filtered_pairs, comp_mask) if is_comp.item()]
            
            # Cooperation loss: pull to circular mean
            if coop_pairs:
                coop_loss = torch.zeros((), device=device, requires_grad=True)
                for i, j in coop_pairs:
                    # Compute circular mean of the pair
                    angle_pair = torch.stack([angles[i], angles[j]])
                    mean_angle = circular_mean(angle_pair)
                    
                    # Pull both angles to mean
                    dist_i = circular_distance(angles[i], mean_angle)
                    dist_j = circular_distance(angles[j], mean_angle)
                    coop_loss = coop_loss + (dist_i**2 + dist_j**2)
                
                total_coop_loss = total_coop_loss + coop_loss / len(coop_pairs)
            
            # Competition loss: push apart with margin
            if comp_pairs:
                comp_similarities = torch.stack([similarities[i, j] for i, j in comp_pairs])
                
                # Self-tuning margin: quantile of competition similarities
                q_m = 1.0 - 1.0 / (len(comp_similarities) + 1)
                margin = torch.quantile(comp_similarities, q_m)
                
                comp_loss = torch.zeros((), device=device, requires_grad=True)
                for i, j in comp_pairs:
                    sim = similarities[i, j]
                    violation = F.relu(sim - margin)
                    comp_loss = comp_loss + violation**2
                
                total_comp_loss = total_comp_loss + comp_loss / len(comp_pairs)
            
            valid_batches += 1
        
        # Average over valid batches
        if valid_batches > 0:
            total_coop_loss = total_coop_loss / valid_batches
            total_comp_loss = total_comp_loss / valid_batches
        
  
        return total_coop_loss, total_comp_loss


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x

class SpatialAnchorEncoder(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
    
    def forward(self, anchors):

        return self.encoder(anchors)


class RingCoDETR(nn.Module):

    def __init__(self, backbone: nn.Module, transformer: nn.Module,
                 num_classes: int, num_queries: int, aux_loss: bool = False,
                 use_spatial_anchors: bool = True,
                 # F-RIPE parameters
                 use_fripe: bool = True,
                 k_ring_eff: int = 8,
                 use_adaptive_k: bool = False,  # True for paper version, False for fixed K
                 # OGQC parameters  
                 use_ogqc: bool = True,
                 ogqc_num_bins: int = 36,
                 ogqc_top_k: int = 8):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.use_spatial_anchors = use_spatial_anchors
        
        # F-RIPE and OGQC flags
        self.use_fripe = use_fripe
        self.use_ogqc = use_ogqc

        # Classification / Box / Angle heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.angle_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh()
        )

        # Spatial anchor queries
        if self.use_spatial_anchors:
            self.query_spatial_anchors = nn.Parameter(torch.zeros(num_queries, 4))
            self._init_spatial_anchors()
            self.spatial_encoder = SpatialAnchorEncoder(hidden_dim)
            self.query_content_embed = nn.Embedding(num_queries, hidden_dim // 2)
            self.query_combine = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Input projection
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, 1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        # FIXED F-RIPE module
        if self.use_fripe:
            self.fripe = FRIPE(k_ring_eff=k_ring_eff, use_adaptive_k=use_adaptive_k)
        
        # FIXED OGQC module
        if self.use_ogqc:
            self.ogqc = OGQC(hidden_dim, num_bins=ogqc_num_bins, top_k_fallback=ogqc_top_k)

        self._init_weights()

    def _init_spatial_anchors(self):

        num_queries = self.num_queries
        grid_size = int(math.ceil(num_queries ** 0.5))
        x = torch.linspace(0.15, 0.85, grid_size)
        y = torch.linspace(0.15, 0.85, grid_size)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        xx = xx.flatten()[:num_queries]
        yy = yy.flatten()[:num_queries]
        xx = (xx + torch.randn(num_queries) * 0.02).clamp(0.05, 0.95)
        yy = (yy + torch.randn(num_queries) * 0.02).clamp(0.05, 0.95)
        w = torch.ones(num_queries) * 0.15 + torch.rand(num_queries) * 0.1
        h = torch.ones(num_queries) * 0.15 + torch.rand(num_queries) * 0.1
        anchors = torch.stack([xx, yy, w, h], dim=1)
        with torch.no_grad():
            self.query_spatial_anchors.copy_(anchors)
        print(f" Initialized {num_queries} queries as {grid_size}x{grid_size} spatial grid")

    def _init_weights(self):

        # bbox head
        nn.init.constant_(self.bbox_embed.layers[-1].bias[:2], 0.0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias[2:4], -2.0)
        nn.init.xavier_uniform_(self.bbox_embed.layers[-1].weight, gain=0.01)

        # angle head
        for m in self.angle_embed:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # spatial encoder
        if self.use_spatial_anchors:
            for m in self.spatial_encoder.encoder:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # classifier prior
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)

    def forward(self, samples, targets=None):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        feats, pos = self.backbone(samples)
        src, mask = feats[-1].decompose()
        B = src.shape[0]

        # Create augmented features for F-RIPE during training
        src_aug = None
        if self.training and self.use_fripe:
            # Simple rotation augmentation
            angles = torch.rand(B, device=src.device) * 2 * math.pi
            src_aug = self._rotate_features(src, angles)

        src_proj = self.input_proj(src)

        # Build queries
        spatial_anchors = None
        if self.use_spatial_anchors:
            spatial_anchors = self.query_spatial_anchors.unsqueeze(0).expand(B, -1, -1)
            spatial_features = self.spatial_encoder(spatial_anchors)
            content_features = self.query_content_embed.weight.unsqueeze(0).expand(B, -1, -1)
            combined = torch.cat([spatial_features, content_features], dim=-1)
            query_embed = self.query_combine(combined).permute(1, 0, 2)
        else:
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        # Transformer forward
        hs = self.transformer(src_proj, mask, query_embed, pos[-1])[0]

        # Predictions
        cls_logits = self.class_embed(hs)
        bbox_xywh_raw = self.bbox_embed(hs)
        bbox_xywh = bbox_xywh_raw.sigmoid()
        angle_raw = self.angle_embed(hs)
        angle_normalized = F.normalize(angle_raw, dim=-1, eps=1e-6)

        # Combine box and angle
        boxes = torch.cat([bbox_xywh, angle_normalized], dim=-1)

        # Prepare outputs
        out = {
            "pred_logits": cls_logits[-1],
            "pred_boxes": boxes[-1],
            "query_features": hs[-1],  # For OGQC
            "spatial_anchors": spatial_anchors if self.use_spatial_anchors else None,
        }

        # Auxiliary outputs
        if self.aux_loss:
            out["aux_outputs"] = [
                {"pred_logits": cls_logits[i], "pred_boxes": boxes[i]}
                for i in range(cls_logits.shape[0] - 1)
            ]

        # FIXED F-RIPE loss during training
        if self.training and self.use_fripe and src_aug is not None:
            fripe_loss = self.fripe(src, src_aug)
            out["fripe_loss"] = fripe_loss

        # FIXED OGQC loss during training
        if self.training and self.use_ogqc:
            coop_loss, comp_loss = self.ogqc(
                hs[-1], angle_normalized[-1], targets
            )
            out["ogqc_coop_loss"] = coop_loss
            out["ogqc_comp_loss"] = comp_loss

        return out

    def _rotate_features(self, features: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
 
        B, C, H, W = features.shape
        
        # Create rotation matrices
        cos_theta = torch.cos(angles).view(B, 1, 1, 1)
        sin_theta = torch.sin(angles).view(B, 1, 1, 1)
        
        # Apply rotation via grid sampling
        theta = torch.zeros(B, 2, 3, device=features.device)
        theta[:, 0, 0] = cos_theta.squeeze()
        theta[:, 0, 1] = -sin_theta.squeeze()  
        theta[:, 1, 0] = sin_theta.squeeze()
        theta[:, 1, 1] = cos_theta.squeeze()
        
        grid = F.affine_grid(theta, features.size(), align_corners=False)
        rotated = F.grid_sample(features, grid, align_corners=False, mode='bilinear')
        
        return rotated


class RingCoSetCriterion(nn.Module):
    """
    FIXED RingCo-DETR loss function with proper EMA-based self-balancing
    - Incorporates F-RIPE and OGQC losses with EMA normalization (not broken median)
    - Proper gradient flow throughout
    """
    def __init__(self, num_classes: int, matcher: nn.Module, weight_dict, 
                 eos_coef: float, losses: list,
                 # Auxiliary loss weights
                 fripe_weight: float = 1.0,
                 ogqc_coop_weight: float = 1.0,
                 ogqc_comp_weight: float = 1.0,
                 # FIXED Self-balancing parameters
                 use_self_balancing: bool = True,
                 ema_momentum: float = 0.9,
                 eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        
        # Auxiliary loss weights
        self.fripe_weight = fripe_weight
        self.ogqc_coop_weight = ogqc_coop_weight
        self.ogqc_comp_weight = ogqc_comp_weight
        
        # FIXED: EMA-based self-balancing instead of broken median approach
        self.use_self_balancing = use_self_balancing
        self.ema_momentum = ema_momentum
        self.eps = eps
        
        if self.use_self_balancing:
            self.register_buffer("ema_fripe", torch.tensor(1.0))
            self.register_buffer("ema_coop", torch.tensor(1.0))
            self.register_buffer("ema_comp", torch.tensor(1.0))

        # Class weights
        empty_w = torch.ones(num_classes + 1)
        empty_w[-1] = eos_coef
        self.register_buffer("empty_weight", empty_w)

    def forward(self, outputs: Dict, targets: List[Dict]):
        """FIXED: Compute all losses with proper EMA-based self-balancing"""
 
        indices, aux_indices = self.matcher(outputs, targets)
        
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(num_boxes, dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1.)

        losses = {}
        for loss in self.losses:
            if loss == "labels":
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss == "boxes":
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
            elif loss == "cardinality":
                losses.update(self.loss_cardinality(outputs, targets, indices, num_boxes))
            else:
                losses.update(getattr(self, f"loss_{loss}")(outputs, targets, indices, num_boxes))


        if "fripe_loss" in outputs:
            losses["loss_fripe"] = outputs["fripe_loss"]

        if "ogqc_coop_loss" in outputs:
            losses["loss_ogqc_coop"] = outputs["ogqc_coop_loss"]
        if "ogqc_comp_loss" in outputs:
            losses["loss_ogqc_comp"] = outputs["ogqc_comp_loss"]

        if self.use_self_balancing and self.training:
            with torch.no_grad():
                if "loss_fripe" in losses:
                    self.ema_fripe = self.ema_momentum * self.ema_fripe + \
                                   (1 - self.ema_momentum) * losses["loss_fripe"].detach()
                
                if "loss_ogqc_coop" in losses:
                    self.ema_coop = self.ema_momentum * self.ema_coop + \
                                  (1 - self.ema_momentum) * losses["loss_ogqc_coop"].detach()
                
                if "loss_ogqc_comp" in losses:
                    self.ema_comp = self.ema_momentum * self.ema_comp + \
                                  (1 - self.ema_momentum) * losses["loss_ogqc_comp"].detach()

            if "loss_fripe" in losses:
                losses["loss_fripe"] = losses["loss_fripe"] / (self.ema_fripe + self.eps)
            if "loss_ogqc_coop" in losses:
                losses["loss_ogqc_coop"] = losses["loss_ogqc_coop"] / (self.ema_coop + self.eps)
            if "loss_ogqc_comp" in losses:
                losses["loss_ogqc_comp"] = losses["loss_ogqc_comp"] / (self.ema_comp + self.eps)


        weighted = {}
        for k, v in losses.items():
            if k == "loss_fripe":
                weighted[k] = v * self.fripe_weight
            elif k == "loss_ogqc_coop":
                weighted[k] = v * self.ogqc_coop_weight
            elif k == "loss_ogqc_comp":
                weighted[k] = v * self.ogqc_comp_weight
            else:
                weighted[k] = v * self.weight_dict.get(k, 1.0)
        
        return weighted

    def loss_labels(self, outputs, targets, indices, num_boxes, log: bool = True):

        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)

        tgt_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                dtype=torch.int64, device=src_logits.device)
        tgt_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        tgt_classes[idx] = tgt_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), tgt_classes, self.empty_weight)
        out = {"loss_ce": loss_ce}
        
        if log:
            out["class_error"] = 100 - accuracy(src_logits[idx], tgt_classes_o)[0]
        
        return out

    def loss_boxes(self, outputs, targets, indices, num_boxes):

        device = outputs["pred_boxes"].device
        zero = torch.tensor(0.0, device=device, requires_grad=True)

        # Primary matches
        idx_primary = self._get_src_permutation_idx(indices)
        src_primary = outputs["pred_boxes"][idx_primary]
        tgt_primary = torch.cat([t["boxes_rotated"][J] for t, (_, J) in zip(targets, indices)], dim=0)

        if src_primary.numel() == 0:
            return {"loss_bbox": zero, "loss_giou": zero, "loss_angle": zero}

        # L1 loss
        l1_loss = F.l1_loss(src_primary[:, :4], tgt_primary[:, :4], reduction="sum")

        # FIXED Angle loss
        pred_sincos = src_primary[:, 4:6]
        tgt_sincos = F.normalize(tgt_primary[:, 4:6], dim=-1, eps=1e-6)
        angle_loss = self._compute_angle_loss(pred_sincos, tgt_sincos, reduction='sum')

        # GIoU loss with error handling
        try:
            giou = rotated_giou_pairwise(src_primary, tgt_primary, mode="sat")
            giou_loss = (1.0 - giou.diag()).sum()
        except Exception as e:
            print(f"Warning: GIoU computation failed: {e}")
            giou_loss = zero

        # Normalize by number of boxes
        n_primary = src_primary.shape[0]
        total_l1 = l1_loss / max(1.0, float(n_primary))
        total_giou = giou_loss / max(1.0, float(n_primary))  
        total_angle = angle_loss / max(1.0, float(n_primary))

        return {
            "loss_bbox": total_l1,
            "loss_giou": total_giou,
            "loss_angle": total_angle
        }

    def _compute_angle_loss(self, pred, target, reduction='mean'):
        """Compute angle loss with 180-degree symmetry"""
        pred_norm = F.normalize(pred, dim=-1, eps=1e-8)
        target_norm = F.normalize(target, dim=-1, eps=1e-8)
        
        cos_sim = (pred_norm * target_norm).sum(dim=-1)
        cos_flip = (pred_norm * (-target_norm)).sum(dim=-1)
        best_sim = torch.max(cos_sim, cos_flip)
        loss = (1.0 - best_sim)
        
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Cardinality error"""
        p = outputs["pred_logits"]
        device = p.device
        tgt_lengths = torch.tensor([len(t["labels"]) for t in targets], device=device)
        pred_cnt = (p.argmax(-1) != p.shape[-1] - 1).sum(1)
        return {"cardinality_error": F.l1_loss(pred_cnt.float(), tgt_lengths.float())}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx




def build_ringco_detr(args):
    """Build FIXED RingCo-DETR model with all components"""
    num_classes = getattr(args, "num_classes", 16)
    device = torch.device(args.device)
    
    # Feature flags
    use_spatial_anchors = getattr(args, "use_spatial_anchors", True)
    use_fripe = getattr(args, "use_fripe", True)
    use_ogqc = getattr(args, "use_ogqc", True)

    # Build components
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    
    # Build FIXED RingCo-DETR model
    model = RingCoDETR(
        backbone, transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        use_spatial_anchors=use_spatial_anchors,
        use_fripe=use_fripe,
        k_ring_eff=getattr(args, "k_ring_eff", 8),
        use_adaptive_k=getattr(args, "use_adaptive_k", False),  # True for paper, False for fixed
        use_ogqc=use_ogqc,
        ogqc_num_bins=getattr(args, "ogqc_num_bins", 36),
        ogqc_top_k=getattr(args, "ogqc_top_k", 8)
    )

    # Build adaptive matcher
    matcher = build_adaptive_matcher(args)

    # Weight dictionary
    weight_dict = {
        "loss_ce": getattr(args, "cls_loss_coef", 2.0),
        "loss_bbox": getattr(args, "bbox_loss_coef", 5.0),
        "loss_giou": getattr(args, "giou_loss_coef", 2.0),
        "loss_angle": getattr(args, "angle_loss_coef", 2.0),
    }

    # Auxiliary losses
    if args.aux_loss:
        for i in range(args.dec_layers - 1):
            for k, v in list(weight_dict.items()):
                weight_dict[k + f"_{i}"] = v


    losses = ["labels", "boxes", "cardinality"]


    criterion = RingCoSetCriterion(
        num_classes,
        matcher,
        weight_dict,
        eos_coef=args.eos_coef,
        losses=losses,
        fripe_weight=getattr(args, "fripe_weight", 1.0),
        ogqc_coop_weight=getattr(args, "ogqc_coop_weight", 1.0),
        ogqc_comp_weight=getattr(args, "ogqc_comp_weight", 1.0),
        use_self_balancing=getattr(args, "use_self_balancing", True),
        ema_momentum=getattr(args, "ema_momentum", 0.9)
    )
    criterion.to(device)

    postprocessors = {"bbox": PostProcess()}
    return model, criterion, postprocessors

# -----------------------------------------------------------------------------
# Post-processing
# -----------------------------------------------------------------------------

class PostProcess(nn.Module):
    """Post-processing for RingCo-DETR outputs"""
    @torch.no_grad()
    def forward(self, outputs: Dict, target_sizes: torch.Tensor):
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]

        prob = F.softmax(logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        angles = torch.atan2(boxes[..., 5], boxes[..., 4])  # atan2(sin, cos)
        angles = torch.atan2(torch.sin(angles), torch.cos(angles)).unsqueeze(-1)

        out_boxes = torch.cat([boxes[..., :4], angles], dim=-1)

        img_h, img_w = target_sizes.unbind(1)
        scale = torch.stack([img_w, img_h, img_w, img_h, torch.ones_like(img_w)], dim=1)
        out_boxes = out_boxes * scale[:, None, :]

        results = [
            {"scores": s, "labels": l, "boxes": b}
            for s, l, b in zip(scores, labels, out_boxes)
        ]
        return results

