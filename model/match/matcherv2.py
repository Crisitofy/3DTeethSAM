import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class TeethHungarianMatcher(nn.Module):
    """牙齿匈牙利匹配器 - 针对 num_queries=16 的简化版本"""
    
    def __init__(self, cost_class=1.0, cost_mask=5.0, cost_dice=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "所有代价权重不能全为0"
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        执行匹配计算
        
        Args:
            outputs: 包含预测的字典:
                - pred_logits: [B, 16, 16] 分类logits
                - pred_masks: [B, 17, H, W] 掩码logits
                
            targets: 包含目标的字典:
                - labels: [B, 16] 目标类别
                - masks: [B, 17, H, W] 目标掩码
                - valid_masks: [B, 16] 有效掩码标志
                
        Returns:
            indices: 匹配索引列表
            metrics: 匹配质量指标
        """
        bs = outputs["pred_logits"].shape[0]
        device = outputs["pred_logits"].device
        
        # 提取预测数据
        out_prob = outputs["pred_logits"].softmax(-1)  # [B, 16, 16]
        out_mask = outputs["pred_masks"][:, 1:17]  # [B, 16, H, W] - 不包括背景
        
        # 提取目标数据
        tgt_ids = targets["labels"]  # [B, 16]
        tgt_mask = targets["masks"][:, 1:17]  # [B, 16, H, W] - 不包括背景
        valid_masks = targets["valid_masks"]  # [B, 16]
        
        indices = []
        total_iou = 0
        total_matches = 0
        
        # 对每个批次进行处理
        for b in range(bs):
            # 提取当前批次的预测和目标数据
            b_out_prob = out_prob[b]  # [16, 16]
            b_out_mask = out_mask[b]  # [16, H, W]
            b_tgt_ids = tgt_ids[b]    # [16]
            b_tgt_mask = tgt_mask[b]  # [16, H, W]
            # print(f"tgt_mask:{tgt_mask.shape}")
            b_valid = valid_masks[b]  # [16]
            
            # 筛选有效目标
            valid_indices = torch.nonzero(b_valid).squeeze(-1)  # [num_valid]
            
            if len(valid_indices) == 0:
                # 没有有效目标，返回空匹配
                indices.append((
                    torch.tensor([], dtype=torch.int64, device=device),
                    torch.tensor([], dtype=torch.int64, device=device)
                ))
                continue
            
            # 提取有效目标数据
            valid_tgt_ids = b_tgt_ids[valid_indices]     # [num_valid]
            valid_tgt_mask = b_tgt_mask[valid_indices]   # [num_valid, H, W]
            
            # 1. 计算分类代价 - 提取每个预测对应有效目标的分类概率
            cost_class = -b_out_prob[:, valid_tgt_ids]  # [16, num_valid]
            
            # 2. 计算掩码IoU代价
            # 将连续值掩码转为二值掩码进行IoU计算
            pred_mask = torch.sigmoid(b_out_mask)  # [16, H, W]
            
            # 重塑掩码用于批量计算
            pred_mask_unsqueeze = pred_mask.unsqueeze(1)  # [16, 1, H, W]
            tgt_mask_unsqueeze = valid_tgt_mask.unsqueeze(0)  # [1, num_valid, H, W]
            
            # 高效计算交集和并集
            intersection = (pred_mask_unsqueeze * tgt_mask_unsqueeze).flatten(2).sum(-1)  # [16, num_valid]
            
            pred_area = pred_mask_unsqueeze.flatten(2).sum(-1)  # [16, 1]
            tgt_area = tgt_mask_unsqueeze.flatten(2).sum(-1)    # [1, num_valid]
            
            union = pred_area + tgt_area - intersection  # [16, num_valid]
            
            # 计算IoU和掩码代价
            iou = intersection / (union + 1e-6)  # [16, num_valid]
            cost_mask = 1 - iou  # [16, num_valid]
            
            # 3. 计算Dice系数代价
            dice_num = 2 * intersection  # [16, num_valid]
            dice_den = pred_area + tgt_area  # [16, num_valid]
            dice_score = dice_num / (dice_den + 1e-6)  # [16, num_valid]
            cost_dice = 1 - dice_score  # [16, num_valid]
            
            # 4. 计算最终代价矩阵
            C = (
                self.cost_class * cost_class +
                self.cost_mask * cost_mask +
                self.cost_dice * cost_dice
            )
            
            # 使用匈牙利算法计算最优匹配
            C_cpu = C.cpu()
            indices_i, indices_j = linear_sum_assignment(C_cpu.numpy())
            
            # 转换回设备上的张量
            indices_i = torch.as_tensor(indices_i, dtype=torch.int64, device=device)
            indices_j = torch.as_tensor(indices_j, dtype=torch.int64, device=device)
            
            # 收集匹配质量统计
            if len(indices_i) > 0:
                match_ious = iou[indices_i, indices_j]
                total_iou += match_ious.sum().item()
                total_matches += len(indices_i)
            
            # 将匹配索引添加到结果列表
            indices.append((indices_i, valid_indices[indices_j]))
        
        # 计算匹配质量指标
        avg_iou = total_iou / max(total_matches, 1)
        metrics = {
            'avg_iou': avg_iou,
            'total_matches': total_matches
        }
        
        return indices, metrics