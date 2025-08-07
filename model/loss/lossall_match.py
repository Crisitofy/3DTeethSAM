import torch
import torch.nn as nn
import torch.nn.functional as F

from model.match.matcherv2 import TeethHungarianMatcher

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
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=device, dtype=torch.float32)
            elif loss.device != device:
                loss = loss.to(device)
            loss_values.append(loss)

        log_vars = torch.clamp(self.log_vars, min=-10, max=10)

        weights = 0.5 / torch.exp(log_vars)
        stacked_losses = torch.stack(loss_values)

        weighted_losses = weights * stacked_losses + 0.5 * log_vars
        total_weighted_loss = weighted_losses.sum()

        weight_dict = {f'weight_{key}': w.item() for key, w in zip(self.loss_keys, weights)}

        return total_weighted_loss, weight_dict

class EndToEndTeethLoss(nn.Module):
    def __init__(self, 
                 stage1_keys=['bce_apg', 'dice_apg', 'conf_apg'],
                 stage2_keys=['ce_refine', 'dice_refine', 'boundary_refine'],
                 cls_loss_weight=1.0,
                 cost_class=1.0, 
                 cost_mask=2.0, 
                 cost_dice=1.0,
                 label_smoothing=0.1):
        super().__init__()
        self.cls_loss_weight = cls_loss_weight
        self.label_smoothing = label_smoothing

        # Initialize matcher
        self.matcher = TeethHungarianMatcher(
            cost_class=cost_class,
            cost_mask=cost_mask,
            cost_dice=cost_dice
        )
        
        # Initialize boundary loss
        self.boundary_loss = DifferentiableBoundaryLoss()
        
        # Initialize uncertainty weighting
        self.stage1_keys = stage1_keys
        self.stage2_keys = stage2_keys
        self.uncertainty_weighting_apg = UncertaintyWeightedLoss(loss_keys=self.stage1_keys)
        self.uncertainty_weighting_refine = UncertaintyWeightedLoss(loss_keys=self.stage2_keys)
        

    def _get_matched_indices(self, stage1_outputs, gt_data):
        """Get matching indices."""
        matcher_outputs = {
            "pred_logits": stage1_outputs['class_logits'],
            "pred_masks": stage1_outputs['sam_masks']
        }
        
        matcher_targets = {
            "labels": gt_data['gt_tooth_classes'],
            "masks": gt_data['gt_masks'],
            "valid_masks": gt_data['valid_masks']
        }
        
        return self.matcher(matcher_outputs, matcher_targets)
    
    def _generate_permuted_masks(self, gt_masks, indices):
        """Generate permuted masks based on matching results."""
        B, C, H, W = gt_masks.shape
        device = gt_masks.device
        
        permuted_masks = torch.zeros_like(gt_masks)
        permuted_masks[:, 0] = gt_masks[:, 0]
        
        for b in range(B):
            pred_indices, gt_indices = indices[b]
            
            if len(pred_indices) == 0:
                continue
                
            for i, j in zip(pred_indices, gt_indices):
                # Assign the (j+1)-th GT mask channel to the (i+1)-th permuted channel
                # +1 is because index 0 is the background channel
                permuted_masks[b, i+1] = gt_masks[b, j+1]
        
        return permuted_masks
    
    def _calculate_stage1_loss(self, stage1_outputs, gt_data, indices):
        """Calculate Stage 1 loss."""
        B = stage1_outputs['sam_masks'].shape[0]
        device = stage1_outputs['sam_masks'].device
        
        teeth_pred_logits = stage1_outputs['sam_masks'][:, 1:]
        teeth_gt = gt_data['gt_masks'][:, 1:]
        confidence = stage1_outputs['confidence']
        
        bce_losses, dice_losses, conf_losses = [], [], []
        
        for b in range(B):
            pred_indices, gt_indices = indices[b]
            
            if len(pred_indices) == 0:
                continue
            
            # Matched mask loss
            b_pred_logits = teeth_pred_logits[b, pred_indices]
            b_gt_masks = teeth_gt[b, gt_indices]
            
            bce_losses.append(F.binary_cross_entropy_with_logits(
                b_pred_logits.flatten(), b_gt_masks.float().flatten(), reduction='mean'
            ))
            
            b_pred_prob = torch.sigmoid(b_pred_logits)
            intersection = (b_pred_prob * b_gt_masks).flatten(1).sum(-1)
            pred_sum = b_pred_prob.flatten(1).sum(-1)
            gt_sum = b_gt_masks.flatten(1).sum(-1)
            b_dice = 1 - (2 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)
            dice_losses.append(b_dice.mean())
            
            # Confidence loss
            target_conf = torch.zeros_like(confidence[b])
            target_conf[pred_indices] = 1.0
            conf_losses.append(F.binary_cross_entropy_with_logits(
                confidence[b], target_conf, reduction='mean'
            ))
        
        return {
            'bce_apg': torch.stack(bce_losses).mean() if bce_losses else torch.tensor(0.0, device=device),
            'dice_apg': torch.stack(dice_losses).mean() if dice_losses else torch.tensor(0.0, device=device),
            'conf_apg': torch.stack(conf_losses).mean() if conf_losses else torch.tensor(0.0, device=device)
        }
    
    def _calculate_stage2_loss(self, stage2_outputs, gt_data, indices):
        """Calculate Stage 2 loss."""
        permuted_masks = self._generate_permuted_masks(gt_data['gt_masks'], indices)
        refined_logits = stage2_outputs['refined_logits']
        gt_indices = torch.argmax(permuted_masks, dim=1).long()
        
        # CE loss
        ce_loss = F.cross_entropy(refined_logits, gt_indices, reduction='mean')
        
        # Dice loss
        pred_probs = F.softmax(refined_logits, dim=1)
        intersection = (pred_probs * permuted_masks).sum(dim=(2, 3))
        pred_sum = pred_probs.sum(dim=(2, 3))
        gt_sum = permuted_masks.sum(dim=(2, 3))
        dice_scores = (2. * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)
        dice_loss = (1.0 - dice_scores).mean()
        
        # Boundary loss
        boundary_loss = self.boundary_loss(pred_probs, permuted_masks)
        
        return {
            'ce_refine': ce_loss,
            'dice_refine': dice_loss,
            'boundary_refine': boundary_loss
        }
    
    def _calculate_classification_loss(self, class_logits, gt_tooth_classes, indices):
        """Improved classification loss - provides supervision for all queries."""
        B, device = class_logits.shape[0], class_logits.device
        cls_losses, correct_sum, total_samples = [], 0, 0
        
        # Extend class space to include a background class (index 16)
        extended_class_logits = F.pad(class_logits, (0, 1), value=0)
        
        for b in range(B):
            pred_indices, gt_indices = indices[b]
            
            # Default target is background class (16)
            target_classes = torch.full((16,), 16, dtype=torch.long, device=device)
            
            if len(pred_indices) > 0:
                # Set correct class for matched queries
                target_classes[pred_indices] = gt_tooth_classes[b, gt_indices].long()
                
                # Calculate accuracy on matched queries
                pred_classes = torch.argmax(class_logits[b, pred_indices], dim=1)
                correct_sum += (pred_classes == gt_tooth_classes[b, gt_indices]).sum().item()
                total_samples += len(pred_indices)
            
            b_loss = F.cross_entropy(
                extended_class_logits[b], target_classes,
                reduction='mean', label_smoothing=self.label_smoothing
            )
            cls_losses.append(b_loss)
        
        cls_loss = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device)
        cls_accuracy = correct_sum / max(total_samples, 1)
        
        return cls_loss, torch.tensor(cls_accuracy, device=device)
    
    def forward(self, stage1_outputs, stage2_outputs, gt_data):
        """Forward pass for loss calculation."""
        indices, _ = self._get_matched_indices(stage1_outputs, gt_data)
        
        losses_apg = self._calculate_stage1_loss(stage1_outputs, gt_data, indices)
        weighted_loss_apg, weights_apg = self.uncertainty_weighting_apg(losses_apg)
        
        losses_refine = self._calculate_stage2_loss(stage2_outputs, gt_data, indices)
        weighted_loss_refine, weights_refine = self.uncertainty_weighting_refine(losses_refine)
        
        cls_loss, cls_accuracy = self._calculate_classification_loss(
            stage1_outputs['class_logits'], gt_data['gt_tooth_classes'], indices
        )
        
        seg_total_loss = weighted_loss_apg + weighted_loss_refine
        total_loss = self.cls_loss_weight * cls_loss + seg_total_loss

        # Prepare report dictionary
        all_losses = {**losses_apg, **losses_refine}
        all_weights = {**weights_apg, **weights_refine}
        report_losses = {k: v.item() for k, v in all_losses.items()}
        report_dict = {**report_losses, **all_weights,
                       'seg_total_loss': seg_total_loss.item(),
                       'total_loss': total_loss.item(),
                       'cls_loss_weight': self.cls_loss_weight,
                       'cls_loss': cls_loss.item(),
                       'cls_accuracy': cls_accuracy.item()}
        
        return total_loss, report_dict, indices