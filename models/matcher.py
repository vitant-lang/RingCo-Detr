

import math
import numpy as np
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment
from util.box_ops import rotated_giou_pairwise, rotated_iou_pairwise


def _quantile_norm(x: torch.Tensor, q: float = 0.95, eps: float = 1e-6) -> torch.Tensor:

    if x.numel() == 0:
        return x
    
    qv = torch.quantile(x.flatten(), q)
    scale = (qv + eps)
    y = (x / scale).clamp_(0.0, 2.0)  # Cap at 2.0 to prevent extreme values
    return y


class AdaptiveHungarianMatcher(nn.Module):

    
    def __init__(self,
                 # Base cost weights (will be adaptively scaled)
                 cost_class: float = 1.0,
                 cost_bbox: float = 5.0, 
                 cost_angle: float = 1.0,
                 cost_iou: float = 2.0,
                 
                 # EMA adaptive scaling parameters
                 use_adaptive: bool = True,
                 ema_alpha: float = 0.95,
                 target_mean: float = 0.6,
                 min_scale: float = 0.25,
                 max_scale: float = 4.0,
                 
                 # Quantile normalization
                 use_quantile_norm: bool = True,
                 quantile_q: float = 0.95,
                 
                 # IoU-ring auxiliary parameters
                 use_auxiliary: bool = True,
                 max_aux_per_target: int = 3,
                 aux_temp_scale: float = 0.5,
                 aux_weight: float = 0.3,
                 
                 # Ring bounds (self-tuning)
                 ring_min_base: float = 0.3,
                 ring_max_base: float = 0.8,
                 ring_ema_alpha: float = 0.9,
                 
                 # Spatial diversity
                 num_angle_bins: int = 36,  # 10° per bin
                 min_spatial_distance: float = 0.1,
                 
                 # Misc
                 eps: float = 1e-6,
                 log_prob: float = 0.01):
        
        super().__init__()
        
        # Base weights
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox  
        self.cost_angle = cost_angle
        self.cost_iou = cost_iou
        
        # Adaptive scaling
        self.use_adaptive = use_adaptive
        self.ema_alpha = ema_alpha
        self.target_mean = target_mean
        self.min_scale = min_scale
        self.max_scale = max_scale
        
        # Normalization
        self.use_quantile_norm = use_quantile_norm
        self.quantile_q = quantile_q
        
        # Auxiliary assignment
        self.use_auxiliary = use_auxiliary
        self.max_aux_per_target = max_aux_per_target
        self.aux_temp_scale = aux_temp_scale
        self.aux_weight = aux_weight
        
        # Ring parameters
        self.ring_min_base = ring_min_base
        self.ring_max_base = ring_max_base
        self.ring_ema_alpha = ring_ema_alpha
        
        # Spatial diversity
        self.num_angle_bins = num_angle_bins
        self.min_spatial_distance = min_spatial_distance
        
        self.eps = eps
        self.log_prob = log_prob
        
        # EMA buffers for cost component scaling
        self.register_buffer("ema_class", torch.tensor(target_mean))
        self.register_buffer("ema_bbox", torch.tensor(target_mean))
        self.register_buffer("ema_angle", torch.tensor(target_mean))
        self.register_buffer("ema_iou", torch.tensor(target_mean))
        
        # EMA buffers for ring bounds per image (init as tensors)
        self.register_buffer("ring_max_ema", torch.tensor(ring_max_base))

    def _update_ema_scales(self, mean_class: float, mean_bbox: float, 
                          mean_angle: float, mean_iou: float) -> Tuple[float, float, float, float]:
    
        if not self.use_adaptive:
            return 1.0, 1.0, 1.0, 1.0
        
        alpha = self.ema_alpha
        
        # Update EMAs
        self.ema_class = (1 - alpha) * self.ema_class + alpha * mean_class
        self.ema_bbox = (1 - alpha) * self.ema_bbox + alpha * mean_bbox  
        self.ema_angle = (1 - alpha) * self.ema_angle + alpha * mean_angle
        self.ema_iou = (1 - alpha) * self.ema_iou + alpha * mean_iou
        
        # Compute adaptive scales (target_mean / ema_mean)
        scale_class = self.target_mean / (self.ema_class + self.eps)
        scale_bbox = self.target_mean / (self.ema_bbox + self.eps)
        scale_angle = self.target_mean / (self.ema_angle + self.eps) 
        scale_iou = self.target_mean / (self.ema_iou + self.eps)
        
        # Clamp to reasonable range
        scale_class = torch.clamp(scale_class, self.min_scale, self.max_scale).item()
        scale_bbox = torch.clamp(scale_bbox, self.min_scale, self.max_scale).item()
        scale_angle = torch.clamp(scale_angle, self.min_scale, self.max_scale).item()
        scale_iou = torch.clamp(scale_iou, self.min_scale, self.max_scale).item()
        
        return scale_class, scale_bbox, scale_angle, scale_iou

    def _compute_cost_components(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor,
                                target_labels: torch.Tensor, target_boxes: torch.Tensor) -> Dict[str, torch.Tensor]:

        
        # Classification cost: 1 - p(target_class)
        pred_probs = pred_logits.softmax(-1)  # [num_queries, num_classes + 1]
        cost_class = 1 - pred_probs[:, target_labels]  # [num_queries, num_targets]
        
        # Box L1 cost
        cost_bbox = torch.cdist(pred_boxes[:, :4], target_boxes[:, :4], p=1)
        
        # Angle cost (cosine similarity with 180° symmetry)
        pred_angles = F.normalize(pred_boxes[:, 4:6], dim=-1, eps=self.eps)  # [Q, 2]
        target_angles = F.normalize(target_boxes[:, 4:6], dim=-1, eps=self.eps)  # [T, 2] 
        
        # Compute similarities: [Q, T]
        cos_sim = torch.mm(pred_angles, target_angles.t())
        cos_sim_flip = torch.mm(pred_angles, (-target_angles).t())
        best_sim = torch.max(cos_sim, cos_sim_flip)
        cost_angle = 1.0 - best_sim.abs()
        
        # IoU cost (using rotated GIoU)
        try:
            # Compute pairwise GIoU between all pred and target boxes
            giou_matrix = rotated_giou_pairwise(pred_boxes, target_boxes, mode="sat")
            cost_iou = (1.0 - giou_matrix).clamp_(min=0.0, max=2.0)
        except Exception:
            # Fallback to zero IoU cost if GIoU computation fails
            cost_iou = torch.zeros_like(cost_bbox)
        
        return {
            "class": cost_class,
            "bbox": cost_bbox,
            "angle": cost_angle,
            "iou": cost_iou
        }

    def _normalize_costs(self, costs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
   
        normalized = {}
        
        for name, cost in costs.items():
            if self.use_quantile_norm:
                normalized[name] = _quantile_norm(cost, q=self.quantile_q, eps=self.eps)
            else:
                # Simple min-max normalization fallback
                cost_min = cost.min()
                cost_max = cost.max()
                denom = (cost_max - cost_min).clamp_min(self.eps)
                normalized[name] = (cost - cost_min) / denom
        
        return normalized

    def _get_auxiliary_candidates(self, cost_matrix: torch.Tensor, primary_indices: Tuple,
                                 pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
                                 iou_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if not self.use_auxiliary:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
        device = cost_matrix.device
        num_queries, num_targets = cost_matrix.shape
        primary_q_idx, primary_t_idx = primary_indices
        
        aux_q_list = []
        aux_t_list = []
        
        for t_idx in range(num_targets):
            # Skip if this target has no primary assignment
            primary_mask = (primary_t_idx == t_idx)
            if not primary_mask.any():
                continue
            
            primary_q = primary_q_idx[primary_mask][0]  # Get primary query for this target
            target_box = target_boxes[t_idx]
            
            # Compute IoU-based ring bounds (self-tuning)
            target_ious = iou_matrix[:, t_idx]  # IoUs for all queries with this target
            
            # Dynamic ring bounds based on IoU distribution
            iou_median = torch.median(target_ious[target_ious > 0.1])  # Ignore very low IoUs
            iou_75th = torch.quantile(target_ious[target_ious > 0.1], 0.75)
            
            ring_min = max(self.ring_min_base, float(iou_median * 0.8))
            
            # Update ring_max with EMA
            current_ring_max = min(self.ring_max_base, float(iou_75th))
            alpha_ring = self.ring_ema_alpha
            self.ring_max_ema = (1 - alpha_ring) * self.ring_max_ema + alpha_ring * current_ring_max
            ring_max = float(self.ring_max_ema)
            
            # Select candidates in IoU ring
            ring_mask = (target_ious >= ring_min) & (target_ious <= ring_max)
            
            # Exclude primary assignments
            ring_mask[primary_q_idx] = False
            
            if not ring_mask.any():
                continue
            
            candidate_indices = torch.where(ring_mask)[0]
            
            # Spatial diversity: bin candidates by angle around target center
            target_center = target_box[:2]  # [x, y]
            candidate_centers = pred_boxes[candidate_indices, :2]  # [N, 2]
            
            # Compute relative angles
            rel_positions = candidate_centers - target_center.unsqueeze(0)  # [N, 2]
            angles = torch.atan2(rel_positions[:, 1], rel_positions[:, 0])  # [N]
            angles = (angles % (2 * math.pi))  # Normalize to [0, 2π)
            
            # Assign to angle bins
            bin_size = 2 * math.pi / self.num_angle_bins
            bin_indices = (angles / bin_size).long().clamp(0, self.num_angle_bins - 1)
            
            # Select best candidate per bin (lowest cost)
            target_costs = cost_matrix[candidate_indices, t_idx]
            
            bin_survivors = []
            for bin_idx in range(self.num_angle_bins):
                bin_mask = (bin_indices == bin_idx)
                if not bin_mask.any():
                    continue
                
                bin_candidates = candidate_indices[bin_mask]
                bin_costs = target_costs[bin_mask]
                
                # Select lowest cost in this bin
                best_idx = torch.argmin(bin_costs)
                best_candidate = bin_candidates[best_idx]
                bin_survivors.append((best_candidate.item(), bin_costs[best_idx].item()))
            
            # Sort survivors by cost and take top-K
            bin_survivors.sort(key=lambda x: x[1])  # Sort by cost
            
            num_aux = min(self.max_aux_per_target, len(bin_survivors))
            
            for i in range(num_aux):
                q_idx, _ = bin_survivors[i]
                
                # Additional spatial constraint: minimum distance from primary
                primary_center = pred_boxes[primary_q, :2]  
                candidate_center = pred_boxes[q_idx, :2]
                spatial_dist = torch.norm(candidate_center - primary_center)
                
                # Use target diagonal as reference for minimum distance
                target_diag = torch.sqrt(target_box[2]**2 + target_box[3]**2)
                min_dist = self.min_spatial_distance * target_diag
                
                if spatial_dist >= min_dist:
                    aux_q_list.append(q_idx)
                    aux_t_list.append(t_idx)
        
        if aux_q_list:
            aux_q_indices = torch.tensor(aux_q_list, dtype=torch.long, device=device)
            aux_t_indices = torch.tensor(aux_t_list, dtype=torch.long, device=device)
        else:
            aux_q_indices = torch.empty(0, dtype=torch.long, device=device)
            aux_t_indices = torch.empty(0, dtype=torch.long, device=device)
        
        return aux_q_indices, aux_t_indices

    @torch.no_grad()
    def forward(self, outputs: Dict, targets: List[Dict]) -> Tuple[List[Tuple], Optional[List[Tuple]]]:

        batch_size = len(targets)
        pred_logits = outputs["pred_logits"]  # [B, Q, C+1]
        pred_boxes = outputs["pred_boxes"]    # [B, Q, 6] (x,y,w,h,sin,cos)
        
        primary_indices = []
        auxiliary_indices = [] if self.use_auxiliary else None
        
        # Process each image in the batch
        for b in range(batch_size):
            target_labels = targets[b]["labels"]
            target_boxes = targets[b]["boxes_rotated"]  # [T, 6]
            
            num_targets = len(target_labels)
            if num_targets == 0:
                # No targets - empty assignment
                empty_idx = (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
                primary_indices.append(empty_idx)
                if self.use_auxiliary:
                    auxiliary_indices.append(empty_idx)
                continue
            
            # Compute cost components
            costs = self._compute_cost_components(
                pred_logits[b], pred_boxes[b], target_labels, target_boxes
            )
            
            # Compute batch statistics for adaptive scaling
            mean_class = costs["class"].mean().item()
            mean_bbox = costs["bbox"].mean().item() 
            mean_angle = costs["angle"].mean().item()
            mean_iou = costs["iou"].mean().item()
            
            # Update EMA and get adaptive scales
            scale_class, scale_bbox, scale_angle, scale_iou = self._update_ema_scales(
                mean_class, mean_bbox, mean_angle, mean_iou
            )
            
            # Normalize cost components
            normalized_costs = self._normalize_costs(costs)
            
            # Combine with adaptive weights
            final_cost = (
                self.cost_class * scale_class * normalized_costs["class"] +
                self.cost_bbox * scale_bbox * normalized_costs["bbox"] + 
                self.cost_angle * scale_angle * normalized_costs["angle"] +
                self.cost_iou * scale_iou * normalized_costs["iou"]
            )
            
            # Numerical safety
            if not torch.isfinite(final_cost).all():
                final_cost = torch.nan_to_num(final_cost, nan=1.0, posinf=2.0, neginf=0.0)
            
            # Hungarian algorithm for primary assignment
            cost_cpu = final_cost.cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost_cpu)
            
            primary_idx = (
                torch.tensor(row_idx, dtype=torch.long),
                torch.tensor(col_idx, dtype=torch.long)
            )
            primary_indices.append(primary_idx)
            
            # Auxiliary assignment using IoU-ring strategy
            if self.use_auxiliary:
                # Compute IoU matrix for auxiliary selection
                try:
                    iou_matrix = rotated_iou_pairwise(pred_boxes[b], target_boxes)
                except Exception:
                    # Fallback: approximate IoU from GIoU
                    giou_matrix = costs["iou"]
                    iou_matrix = (2.0 - giou_matrix) / 2.0  # Rough approximation
                
                aux_q_idx, aux_t_idx = self._get_auxiliary_candidates(
                    final_cost, primary_idx, pred_boxes[b], target_boxes, iou_matrix
                )
                
                auxiliary_indices.append((aux_q_idx, aux_t_idx))
            
            # Optional: Log matching statistics
            if self.log_prob > 0 and torch.rand(1).item() < self.log_prob:
                print(f"[AdaptiveMatcher] Batch {b}: "
                      f"scales=(cls:{scale_class:.2f}, bbox:{scale_bbox:.2f}, "
                      f"angle:{scale_angle:.2f}, iou:{scale_iou:.2f}) "
                      f"primary={len(primary_idx[0])}, aux={len(aux_q_idx) if self.use_auxiliary else 0}")
        
        return primary_indices, auxiliary_indices

    def get_auxiliary_loss_weights(self, auxiliary_indices: List[Tuple],
                                  cost_matrices: List[torch.Tensor]) -> List[torch.Tensor]:

        if not self.use_auxiliary or auxiliary_indices is None:
            return []
        
        batch_weights = []
        
        for b, (aux_q_idx, aux_t_idx) in enumerate(auxiliary_indices):
            if len(aux_q_idx) == 0:
                batch_weights.append(torch.empty(0))
                continue
            
            # Get costs for auxiliary pairs
            aux_costs = cost_matrices[b][aux_q_idx, aux_t_idx]
            
            # Temperature-scaled softmax weights
            # Lower cost = higher weight
            weights = F.softmax(-aux_costs / self.aux_temp_scale, dim=0)
            batch_weights.append(weights)
        
        return batch_weights


def build_adaptive_matcher(args):

    return AdaptiveHungarianMatcher(
        # Base cost weights
        cost_class=getattr(args, "set_cost_class", 1.0),
        cost_bbox=getattr(args, "set_cost_bbox", 5.0),
        cost_angle=getattr(args, "set_cost_angle", 1.0), 
        cost_iou=getattr(args, "set_cost_giou", 2.0),
        
        # Adaptive scaling parameters  
        use_adaptive=getattr(args, "use_adaptive_matching", True),
        ema_alpha=getattr(args, "matcher_ema_alpha", 0.95),
        target_mean=getattr(args, "matcher_target_mean", 0.6),
        min_scale=getattr(args, "matcher_min_scale", 0.25),
        max_scale=getattr(args, "matcher_max_scale", 4.0),
        
        # Quantile normalization
        use_quantile_norm=getattr(args, "use_quantile_norm", True),
        quantile_q=getattr(args, "quantile_q", 0.95),
        
        # Auxiliary assignment parameters
        use_auxiliary=getattr(args, "use_auxiliary_assignment", True),
        max_aux_per_target=getattr(args, "max_aux_per_target", 3),
        aux_temp_scale=getattr(args, "aux_temp_scale", 0.5),
        aux_weight=getattr(args, "aux_weight", 0.3),
        
        # Ring parameters
        ring_min_base=getattr(args, "ring_min_base", 0.3),
        ring_max_base=getattr(args, "ring_max_base", 0.8),
        ring_ema_alpha=getattr(args, "ring_ema_alpha", 0.9),
        
        # Spatial diversity parameters
        num_angle_bins=getattr(args, "num_angle_bins", 36),
        min_spatial_distance=getattr(args, "min_spatial_distance", 0.1),
        
        # Logging
        log_prob=getattr(args, "matcher_log_prob", 0.01)
    )



class AuxiliaryAwareLoss(nn.Module):

    def __init__(self, base_criterion, aux_weight: float = 0.3):
        super().__init__()
        self.base_criterion = base_criterion
        self.aux_weight = aux_weight
    
    def forward(self, outputs, targets, primary_indices, auxiliary_indices=None):
        """Compute losses including auxiliary supervision"""
        
        # Standard loss computation
        losses = self.base_criterion(outputs, targets, primary_indices)
        
        # Add auxiliary losses if available
        if auxiliary_indices is not None and any(len(aux[0]) > 0 for aux in auxiliary_indices):
            aux_losses = self._compute_auxiliary_losses(
                outputs, targets, primary_indices, auxiliary_indices
            )
            
            # Weight and add auxiliary losses
            for key, value in aux_losses.items():
                aux_key = f"aux_{key}"
                losses[aux_key] = value * self.aux_weight
        
        return losses
    
    def _compute_auxiliary_losses(self, outputs, targets, primary_indices, auxiliary_indices):
        """Compute losses for auxiliary assignments"""
        
        device = outputs["pred_logits"].device
        aux_losses = {}
        
        # Classification loss for auxiliary pairs
        aux_ce_loss = 0.0
        aux_bbox_loss = 0.0
        aux_angle_loss = 0.0
        num_aux_total = 0
        
        for b, (aux_q_idx, aux_t_idx) in enumerate(auxiliary_indices):
            if len(aux_q_idx) == 0:
                continue
            
            # Auxiliary classification targets
            aux_labels = targets[b]["labels"][aux_t_idx]
            aux_logits = outputs["pred_logits"][b][aux_q_idx]
            
            # Cross entropy loss
            aux_ce_loss += F.cross_entropy(aux_logits, aux_labels, reduction='sum')
            
            # Auxiliary box regression
            aux_pred_boxes = outputs["pred_boxes"][b][aux_q_idx]
            aux_target_boxes = targets[b]["boxes_rotated"][aux_t_idx]
            
            # L1 loss
            aux_bbox_loss += F.l1_loss(aux_pred_boxes[:, :4], aux_target_boxes[:, :4], reduction='sum')
            
            # Angle loss
            pred_angles = F.normalize(aux_pred_boxes[:, 4:6], dim=-1, eps=1e-6)
            target_angles = F.normalize(aux_target_boxes[:, 4:6], dim=-1, eps=1e-6)
            
            cos_sim = (pred_angles * target_angles).sum(dim=-1)
            cos_sim_flip = (pred_angles * (-target_angles)).sum(dim=-1)
            best_sim = torch.max(cos_sim, cos_sim_flip)
            aux_angle_loss += (1.0 - best_sim).sum()
            
            num_aux_total += len(aux_q_idx)
        
        # Normalize by number of auxiliary pairs
        if num_aux_total > 0:
            aux_losses["ce"] = aux_ce_loss / num_aux_total
            aux_losses["bbox"] = aux_bbox_loss / num_aux_total  
            aux_losses["angle"] = aux_angle_loss / num_aux_total
        else:
            aux_losses["ce"] = torch.tensor(0.0, device=device)
            aux_losses["bbox"] = torch.tensor(0.0, device=device)
            aux_losses["angle"] = torch.tensor(0.0, device=device)
        
        return aux_losses




def integrate_adaptive_matcher_example(args):

    
    # Build model components
    model, base_criterion, postprocessors = build_ringco_detr(args)
    
    # Build adaptive matcher 
    adaptive_matcher = build_adaptive_matcher(args)
    
    # Wrap criterion to handle auxiliary losses
    aux_aware_criterion = AuxiliaryAwareLoss(
        base_criterion, 
        aux_weight=getattr(args, "aux_weight", 0.3)
    )
    
    return model, adaptive_matcher, aux_aware_criterion, postprocessors


