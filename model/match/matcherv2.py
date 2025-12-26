import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class TeethHungarianMatcher(nn.Module):
    """Tooth Hungarian matcher"""
    
    def __init__(self, cost_class=1.0, cost_mask=5.0, cost_dice=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "At least one cost weight must be non-zero"
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Run matching computation.
        """
        bs = outputs["pred_logits"].shape[0]
        device = outputs["pred_logits"].device
        
        # Extract predicted data
        out_prob = outputs["pred_logits"].softmax(-1)  # [B, 16, 16]
        out_mask = outputs["pred_masks"][:, 1:17]  # [B, 16, H, W] - exclude background
        
        # Extract target data
        tgt_ids = targets["labels"]  # [B, 16]
        tgt_mask = targets["masks"][:, 1:17]  # [B, 16, H, W] - exclude background
        valid_masks = targets["valid_masks"]  # [B, 16]
        
        indices = []
        total_iou = 0
        total_matches = 0
        
        # Process each batch element
        for b in range(bs):
            # Extract current batch predictions and targets
            b_out_prob = out_prob[b]  # [16, 16]
            b_out_mask = out_mask[b]  # [16, H, W]
            b_tgt_ids = tgt_ids[b]    # [16]
            b_tgt_mask = tgt_mask[b]  # [16, H, W]
            # print(f"tgt_mask:{tgt_mask.shape}")
            b_valid = valid_masks[b]  # [16]
            
            # Filter valid targets
            valid_indices = torch.nonzero(b_valid).squeeze(-1)  # [num_valid]
            
            if len(valid_indices) == 0:
                # No valid targets; return empty matches
                indices.append((
                    torch.tensor([], dtype=torch.int64, device=device),
                    torch.tensor([], dtype=torch.int64, device=device)
                ))
                continue
            
            # Extract valid target data
            valid_tgt_ids = b_tgt_ids[valid_indices]     # [num_valid]
            valid_tgt_mask = b_tgt_mask[valid_indices]   # [num_valid, H, W]
            
            # 1. Compute classification cost for valid targets
            cost_class = -b_out_prob[:, valid_tgt_ids]  # [16, num_valid]
            
            # 2. Compute mask IoU cost
            # Convert continuous masks to binary for IoU
            pred_mask = torch.sigmoid(b_out_mask)  # [16, H, W]
            
            # Reshape masks for batch computation
            pred_mask_unsqueeze = pred_mask.unsqueeze(1)  # [16, 1, H, W]
            tgt_mask_unsqueeze = valid_tgt_mask.unsqueeze(0)  # [1, num_valid, H, W]
            
            # Efficiently compute intersection and union
            intersection = (pred_mask_unsqueeze * tgt_mask_unsqueeze).flatten(2).sum(-1)  # [16, num_valid]
            
            pred_area = pred_mask_unsqueeze.flatten(2).sum(-1)  # [16, 1]
            tgt_area = tgt_mask_unsqueeze.flatten(2).sum(-1)    # [1, num_valid]
            
            union = pred_area + tgt_area - intersection  # [16, num_valid]
            
            # Compute IoU and mask cost
            iou = intersection / (union + 1e-6)  # [16, num_valid]
            cost_mask = 1 - iou  # [16, num_valid]
            
            # 3. Compute Dice coefficient cost
            dice_num = 2 * intersection  # [16, num_valid]
            dice_den = pred_area + tgt_area  # [16, num_valid]
            dice_score = dice_num / (dice_den + 1e-6)  # [16, num_valid]
            cost_dice = 1 - dice_score  # [16, num_valid]
            
            # 4. Compute final cost matrix
            C = (
                self.cost_class * cost_class +
                self.cost_mask * cost_mask +
                self.cost_dice * cost_dice
            )
            
            # Use Hungarian algorithm to compute optimal matching
            C_cpu = C.cpu()
            indices_i, indices_j = linear_sum_assignment(C_cpu.numpy())
            
            # Move back to device tensors
            indices_i = torch.as_tensor(indices_i, dtype=torch.int64, device=device)
            indices_j = torch.as_tensor(indices_j, dtype=torch.int64, device=device)
            
            # Collect matching quality stats
            if len(indices_i) > 0:
                match_ious = iou[indices_i, indices_j]
                total_iou += match_ious.sum().item()
                total_matches += len(indices_i)
            
            # Append matched indices to results list
            indices.append((indices_i, valid_indices[indices_j]))
        
        # Compute matching quality metrics
        avg_iou = total_iou / max(total_matches, 1)
        metrics = {
            'avg_iou': avg_iou,
            'total_matches': total_matches
        }
        
        return indices, metrics
