import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class TeethHungarianMatcher(nn.Module):
    """Tooth Hungarian matcher"""
    
    def __init__(self, cost_class=1.0, cost_mask=2.0, cost_dice=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "At least one cost must be non-zero"
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """Matcher using batched tensor operations."""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Extract predicted probabilities and masks
        out_prob = outputs["pred_logits"].softmax(-1)  # [B, num_queries, num_classes]
        out_mask = outputs["pred_masks"]  # [B, num_queries, H, W]
        
        # Extract target labels and masks
        tgt_ids = targets["labels"]  # [B, num_targets]
        tgt_mask = targets["masks"]  # [B, num_targets, H, W]
        valid_masks = targets["valid_masks"]  # [B, num_targets]
        
        indices = []
        
        # Process each batch element
        for b in range(bs):
            # Extract current batch data
            b_out_prob = out_prob[b]  # [num_queries, num_classes]
            b_out_mask = out_mask[b]  # [num_queries, H, W]
            
            # Filter valid targets
            b_valid = valid_masks[b]  # [num_targets]
            valid_indices = torch.nonzero(b_valid).squeeze(1)
            
            if len(valid_indices) == 0:
                # Return empty matches when no valid targets
                indices.append((torch.tensor([], dtype=torch.int64, device=out_prob.device), 
                               torch.tensor([], dtype=torch.int64, device=out_prob.device)))
                continue
            
            # Extract valid target data
            b_tgt_ids_valid = tgt_ids[b, valid_indices]  # [num_valid]
            b_tgt_mask_valid = tgt_mask[b, valid_indices]  # [num_valid, H, W]
            
            # Classification cost
            # [num_queries, 17] -> [num_queries, 16]
            cost_class = -b_out_prob[:, :16][:, b_tgt_ids_valid]
            
            # Format masks for computation
            b_pred_mask = torch.sigmoid(b_out_mask)  # [num_queries, H, W]
            
            # Compute IoU in batch
            pred_mask = b_pred_mask.unsqueeze(1)  # [num_queries, 1, H, W]
            tgt_mask = b_tgt_mask_valid.unsqueeze(0)  # [1, num_valid, H, W]
            
            # Compute intersection and union [num_queries, num_valid, H, W]
            both = pred_mask * tgt_mask
            area_both = both.flatten(2).sum(-1)  # [num_queries, num_valid]
            
            area_pred = b_pred_mask.flatten(1).sum(-1).unsqueeze(1)  # [num_queries, 1]
            area_tgt = b_tgt_mask_valid.flatten(1).sum(-1).unsqueeze(0)  # [1, num_valid]
            
            area_union = area_pred + area_tgt - area_both  # [num_queries, num_valid]
            
            # Compute IoU [num_queries, num_valid]
            iou = area_both / (area_union + 1e-6)
            cost_mask = 1 - iou
            
            # Compute Dice coefficient
            dice_numerator = 2 * area_both  # [num_queries, num_valid]
            dice_denominator = area_pred + area_tgt  # [num_queries, num_valid]
            dice_score = dice_numerator / (dice_denominator + 1e-6)
            cost_dice = 1 - dice_score
            
            # Compute final cost matrix
            C = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
            
            # Use Hungarian algorithm to find optimal matches
            C_cpu = C.cpu()
            indices_i, indices_j = linear_sum_assignment(C_cpu.numpy())
            indices_i = torch.as_tensor(indices_i, dtype=torch.int64, device=C.device)
            indices_j = torch.as_tensor(indices_j, dtype=torch.int64, device=C.device)
            
            # Map matched indices back to original target indices
            indices.append((indices_i, valid_indices[indices_j]))
            
        return indices
