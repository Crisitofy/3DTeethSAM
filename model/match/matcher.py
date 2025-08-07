import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class TeethHungarianMatcher(nn.Module):
    """高效的牙齿匈牙利匹配器，使用张量操作避免循环"""
    
    def __init__(self, cost_class=1.0, cost_mask=2.0, cost_dice=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "所有代价不能为 0"
    
    @torch.no_grad()
    def forward(self, outputs, targets):
        """批量张量操作的匹配器"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # 提取预测的概率和掩码
        out_prob = outputs["pred_logits"].softmax(-1)  # [B, num_queries, num_classes]
        out_mask = outputs["pred_masks"]  # [B, num_queries, H, W]
        
        # 提取目标标签和掩码
        tgt_ids = targets["labels"]  # [B, num_targets]
        tgt_mask = targets["masks"]  # [B, num_targets, H, W]
        valid_masks = targets["valid_masks"]  # [B, num_targets]
        
        indices = []
        
        # 对每个批次进行处理
        for b in range(bs):
            # 提取当前批次数据
            b_out_prob = out_prob[b]  # [num_queries, num_classes]
            b_out_mask = out_mask[b]  # [num_queries, H, W]
            
            # 筛选有效目标
            b_valid = valid_masks[b]  # [num_targets]
            valid_indices = torch.nonzero(b_valid).squeeze(1)
            
            if len(valid_indices) == 0:
                # 没有有效目标时返回空匹配
                indices.append((torch.tensor([], dtype=torch.int64, device=out_prob.device), 
                               torch.tensor([], dtype=torch.int64, device=out_prob.device)))
                continue
            
            # 提取有效目标数据
            b_tgt_ids_valid = tgt_ids[b, valid_indices]  # [num_valid]
            b_tgt_mask_valid = tgt_mask[b, valid_indices]  # [num_valid, H, W]
            
            # 分类代价 - 只考虑16个牙齿类别
            # [num_queries, 17] -> [num_queries, 16]
            cost_class = -b_out_prob[:, :16][:, b_tgt_ids_valid]
            
            # 将掩码格式化为适合计算的形式
            b_pred_mask = torch.sigmoid(b_out_mask)  # [num_queries, H, W]
            
            # 批量计算IoU
            # 将预测和目标掩码重塑为[num_queries, 1, H, W]和[1, num_valid, H, W]
            # 这样可以利用广播进行批量计算
            pred_mask = b_pred_mask.unsqueeze(1)  # [num_queries, 1, H, W]
            tgt_mask = b_tgt_mask_valid.unsqueeze(0)  # [1, num_valid, H, W]
            
            # 计算交集和并集 [num_queries, num_valid, H, W]
            both = pred_mask * tgt_mask
            area_both = both.flatten(2).sum(-1)  # [num_queries, num_valid]
            
            area_pred = b_pred_mask.flatten(1).sum(-1).unsqueeze(1)  # [num_queries, 1]
            area_tgt = b_tgt_mask_valid.flatten(1).sum(-1).unsqueeze(0)  # [1, num_valid]
            
            area_union = area_pred + area_tgt - area_both  # [num_queries, num_valid]
            
            # 计算IoU [num_queries, num_valid]
            iou = area_both / (area_union + 1e-6)
            cost_mask = 1 - iou
            
            # 计算Dice系数
            dice_numerator = 2 * area_both  # [num_queries, num_valid]
            dice_denominator = area_pred + area_tgt  # [num_queries, num_valid]
            dice_score = dice_numerator / (dice_denominator + 1e-6)
            cost_dice = 1 - dice_score
            
            # 计算最终代价矩阵
            C = self.cost_class * cost_class + self.cost_mask * cost_mask + self.cost_dice * cost_dice
            
            # 调用匈牙利算法找出最优匹配
            C_cpu = C.cpu()
            indices_i, indices_j = linear_sum_assignment(C_cpu.numpy())
            indices_i = torch.as_tensor(indices_i, dtype=torch.int64, device=C.device)
            indices_j = torch.as_tensor(indices_j, dtype=torch.int64, device=C.device)
            
            # 将匹配索引映射回原始目标索引
            indices.append((indices_i, valid_indices[indices_j]))
            
        return indices