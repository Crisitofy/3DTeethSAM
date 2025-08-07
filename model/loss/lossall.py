import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableBoundaryLoss(nn.Module):
    """
    A differentiable boundary loss based on spatial gradients (Sobel filters).
    This loss is designed to be more stable than hard-thresholding morphological operations.
    """
    def __init__(self):
        super().__init__()
        # Sobel filters for approximating spatial gradients
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def _get_gradient(self, mask):
        """Calculates spatial gradient using Sobel filters with grouped convolution."""
        B, C, H, W = mask.shape
        device = mask.device
        
        # Prepare kernels for grouped convolution to apply filter channel-wise
        k_x = self.kernel_x.to(device).repeat(C, 1, 1, 1)
        k_y = self.kernel_y.to(device).repeat(C, 1, 1, 1)

        grad_x = F.conv2d(mask, k_x, padding='same', groups=C)
        grad_y = F.conv2d(mask, k_y, padding='same', groups=C)
        return grad_x, grad_y

    def forward(self, pred_probs, target_one_hot):
        """
        Args:
            pred_probs: Predicted probabilities, shape [B, C, H, W]
            target_one_hot: Ground truth one-hot mask, shape [B, C, H, W]
        Returns:
            loss: Boundary loss
        """
        pred_grad_x, pred_grad_y = self._get_gradient(pred_probs)
        target_grad_x, target_grad_y = self._get_gradient(target_one_hot.float())
        
        # Loss is the L1 distance between the gradients
        loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
        return loss

class BoundaryLossVectorized(nn.Module):
    """
    边界损失函数
    """
    def __init__(self, kernel_size=3, boundary_width=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.boundary_width = boundary_width
        # 创建卷积核（全1的方形核），注册为 buffer
        kernel = torch.ones(1, 1, kernel_size, kernel_size)
        self.register_buffer('kernel', kernel) 

    def _get_boundary(self, mask):
        """
        通过形态学操作提取掩码的边界
        Args:
            mask: 形状为 [N, H, W] 或 [N, 1, H, W] 的二值掩码 (N=B*C 或 B)
        Returns:
            boundary: 形状为 [N, H, W] 的边界掩码
        """
        if mask.dim() == 3:
            mask = mask.unsqueeze(1) # 变为 [N, 1, H, W]
        N, _, H, W = mask.shape
        device = mask.device # 获取设备信息

        kernel = self.kernel.to(device) # 确保在正确的设备

        padding = self.kernel_size // 2

        # 膨胀操作 - 扩大区域 (向量化)
        dilated = mask
        for _ in range(self.boundary_width):
            dilated_padded = F.pad(dilated, [padding] * 4, mode='constant', value=0)
            dilated = F.conv2d(dilated_padded, kernel, padding=0) # 移除 groups=N
            dilated = (dilated > 0).float()

        # 腐蚀操作 
        eroded = mask
        kernel_size_square = self.kernel_size * self.kernel_size
        for _ in range(self.boundary_width):
            eroded_padded = F.pad(eroded, [padding] * 4, mode='constant', value=1) # 腐蚀 pad 1
            eroded = F.conv2d(eroded_padded, kernel, padding=0) # 移除 groups=N
            eroded = (eroded >= kernel_size_square).float()

        # 边界是膨胀和腐蚀的差异
        # boundary = dilated - eroded  # [N, 1, H, W],修改
        boundary = torch.clamp(dilated - eroded, min=0.0, max=1.0)

        return boundary.squeeze(1) # [N, H, W]

    def forward(self, pred_probs, target_one_hot, channel_weights=None):
        """
        计算预测和目标边界之间的损失
        Args:
            pred_probs: 预测的概率掩码，形状为 [B, C, H, W]
            target_one_hot: 目标 one-hot 掩码，形状为 [B, C, H, W]
        Returns:
            loss: 平均边界损失 
        """
        B, C, H, W = pred_probs.shape
        device = pred_probs.device

        pred_binary_approx = pred_probs

        # 重塑为 [B*C, H, W] 以进行批处理形态学操作
        pred_flat = pred_binary_approx.reshape(B * C, H, W)
        target_flat = target_one_hot.reshape(B * C, H, W)

        # 提取预测和目标的边界
        pred_boundary = self._get_boundary(pred_flat).reshape(B, C, H, W)
        target_boundary = self._get_boundary(target_flat).reshape(B, C, H, W) # GT 边界是确定的

        # 计算所有通道的损失，不筛选有效通道
        loss_pixel = F.mse_loss(pred_boundary, target_boundary, reduction='none') # [B, C, H, W]
        
        # 对所有通道和空间维度求平均
        loss = loss_pixel.mean()

        return loss

class UncertaintyWeightedLoss(nn.Module):
    """Dynamically weighted loss based on uncertainty (CVPR 2018)."""
    def __init__(self, loss_keys):
        super().__init__()
        self.loss_keys = loss_keys
        self.num_losses = len(loss_keys)
        # Initialize learnable log variance parameters
        self.log_vars = nn.Parameter(torch.zeros(self.num_losses))

    def forward(self, losses_dict):
        """
        Args:
            losses_dict: A dictionary of loss values, where keys are in self.loss_keys.
        """
        device = self.log_vars.device

        loss_values = []
        for key in self.loss_keys:
            loss = losses_dict.get(key, None)
            if loss is None:
                raise KeyError(f"Loss key '{key}' not found in losses_dict.")
            # Ensure it is a tensor on the correct device
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=device, dtype=torch.float32)
            elif loss.device != device:
                loss = loss.to(device)
            loss_values.append(loss)

        log_vars = torch.clamp(self.log_vars, min=-10, max=10)

        # Calculate weight for each loss = 1 / (2 * exp(log_var))
        weights = 0.5 / torch.exp(log_vars)
        stacked_losses = torch.stack(loss_values)

        # Calculate weighted loss + regularization term
        weighted_losses = weights * stacked_losses + 0.5 * log_vars
        total_weighted_loss = weighted_losses.sum()

        # Create a dictionary of weights for logging
        weight_dict = {f'weight_{key}': w.item() for key, w in zip(self.loss_keys, weights)}

        return total_weighted_loss, weight_dict

class EndToEndTeethLoss(nn.Module):
    def __init__(self, stage1_keys=['bce_apg', 'dice_apg', 'conf_apg'],
                 stage2_keys=['ce_refine', 'dice_refine', 'boundary_refine']):
        super().__init__()

        # Stage 1 (APG) related
        self.stage1_keys = stage1_keys
        self.uncertainty_weighting_apg = UncertaintyWeightedLoss(loss_keys=self.stage1_keys)

        # Stage 2 (Refine) related
        self.stage2_keys = stage2_keys
        self.boundary_loss_refine = DifferentiableBoundaryLoss()
        self.uncertainty_weighting_refine = UncertaintyWeightedLoss(loss_keys=self.stage2_keys)

    def _calculate_stage1_loss(self, sam_masks, gt_masks_binary, valid_masks, confidence):
        """Calculates Stage 1 (APG) loss for all teeth."""
        # Clamp logits to prevent numerical instability
        sam_masks = torch.clamp(sam_masks, min=-15, max=15)
        confidence = torch.clamp(confidence, min=-15, max=15)
        
        # Extract tooth masks (excluding background)
        teeth_pred_logits = sam_masks[:, 1:]  # [B, 16, H, W]
        teeth_gt = gt_masks_binary[:, 1:]     # [B, 16, H, W]

        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(
            teeth_pred_logits.reshape(-1),
            teeth_gt.float().reshape(-1),
            reduction='mean'
        )

        # Dice Loss
        teeth_pred_prob = torch.sigmoid(teeth_pred_logits)
        intersection = (teeth_pred_prob * teeth_gt).sum(dim=(2, 3))
        union = teeth_pred_prob.sum(dim=(2, 3)) + teeth_gt.sum(dim=(2, 3))
        dice_scores = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = (1.0 - dice_scores).mean()

        # Confidence Loss
        conf_loss = F.binary_cross_entropy_with_logits(
            confidence.view(-1),
            valid_masks.float().view(-1),
            reduction='mean'
        )

        losses_apg = {
            'bce_apg': bce_loss,
            'dice_apg': dice_loss,
            'conf_apg': conf_loss
        }
        return losses_apg

    def _calculate_stage2_loss(self, refined_logits, gt_masks_one_hot):
        """Calculates Stage 2 (Refine) loss for all tooth channels."""
        # Clamp logits to prevent numerical instability
        refined_logits = torch.clamp(refined_logits, min=-15, max=15)

        # Prepare GT for CE Loss
        gt_indices = torch.argmax(gt_masks_one_hot, dim=1)

        # CE Loss
        ce_loss = F.cross_entropy(refined_logits, gt_indices, reduction='mean')

        # Dice Loss
        pred_probs = F.softmax(refined_logits, dim=1)
        intersection = (pred_probs * gt_masks_one_hot).sum(dim=(2, 3))
        pred_sum = pred_probs.sum(dim=(2, 3))
        gt_sum = gt_masks_one_hot.sum(dim=(2, 3))
        dice_scores = (2. * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)
        dice_loss = (1.0 - dice_scores).mean()

        # Boundary Loss
        boundary_loss = self.boundary_loss_refine(pred_probs, gt_masks_one_hot)

        losses_refine = {
            'ce_refine': ce_loss,
            'dice_refine': dice_loss,
            'boundary_refine': boundary_loss
        }
        return losses_refine

    def forward(self, stage1_outputs, stage2_outputs, gt_data):
        """
        Calculates the end-to-end loss.
        """
        all_losses = {}
        all_weights = {}

        # Calculate Stage 1 loss
        losses_apg = self._calculate_stage1_loss(
            stage1_outputs['sam_masks'],
            gt_data['gt_masks'],
            gt_data['valid_masks'],
            stage1_outputs['confidence']
        )
        weighted_loss_apg, weights_apg = self.uncertainty_weighting_apg(losses_apg)
        all_losses.update(losses_apg)
        all_weights.update(weights_apg)

        # Calculate Stage 2 loss
        losses_refine = self._calculate_stage2_loss(
            stage2_outputs['refined_logits'],
            gt_data['gt_masks']
        )
        weighted_loss_refine, weights_refine = self.uncertainty_weighting_refine(losses_refine)
        all_losses.update(losses_refine)
        all_weights.update(weights_refine)

        # Combine total loss
        total_loss = weighted_loss_apg + weighted_loss_refine
        
        # Prepare dictionary for reporting
        report_losses = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in all_losses.items()}
        report_dict = {**report_losses, **all_weights}
        report_dict['total_loss'] = total_loss.item()

        return total_loss, report_dict