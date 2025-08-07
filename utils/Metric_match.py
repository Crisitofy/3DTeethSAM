import torch
from collections import defaultdict
import logging
from torch.utils.tensorboard import SummaryWriter

class MetricCalculator:
    def __init__(self, config):
        self.config = config
        self.reset()
        
    def reset(self):
        self.batch_metrics = {
            'iou': [],
            'dice': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        self.position_metrics = defaultdict(lambda: defaultdict(list))
        self.sample_counts = 0
        self.total_loss = 0
        self.batch_count = 0
        self.cls_accuracy_sum = 0
        self.cls_accuracy_batches = 0 

    def update(self, pred_masks, gt_masks, tooth_ids=None, loss=None, cls_accuracy=None):
        batch_size, num_teeth = pred_masks.shape[0], pred_masks.shape[1]
        pred_masks = (pred_masks > 0.5).float()
        
        intersection = torch.sum(pred_masks * gt_masks, dim=(2, 3))
        pred_sum = torch.sum(pred_masks, dim=(2, 3))
        gt_sum = torch.sum(gt_masks, dim=(2, 3))
        union = pred_sum + gt_sum - intersection
        
        eps = 1e-6
        iou = (intersection + eps) / (union + eps)
        dice = (2 * intersection + eps) / (pred_sum + gt_sum + eps)
        precision = (intersection + eps) / (pred_sum + eps)
        recall = (intersection + eps) / (gt_sum + eps)
        f1 = (2 * precision * recall + eps) / (precision + recall + eps)
        
        self.batch_metrics['iou'].append(iou)
        self.batch_metrics['dice'].append(dice)
        self.batch_metrics['precision'].append(precision)
        self.batch_metrics['recall'].append(recall)
        self.batch_metrics['f1'].append(f1)

        if cls_accuracy is not None:
            if isinstance(cls_accuracy, torch.Tensor):
                cls_accuracy = cls_accuracy.item()
            self.cls_accuracy_sum += cls_accuracy
            self.cls_accuracy_batches += 1

        for tooth_idx in range(num_teeth):
            for metric_name, metric_tensor in zip(
                ['iou', 'dice', 'precision', 'recall', 'f1'], 
                [iou, dice, precision, recall, f1]
            ):
                self.position_metrics[tooth_idx][metric_name].append(metric_tensor[:, tooth_idx])
        
        self.sample_counts += batch_size
        if loss is not None:
            self.total_loss += loss.item()
            self.batch_count += 1
    
    def compute(self):
        """Compute final metrics."""
        avg_loss = self.total_loss / max(1, self.batch_count)
        final_metrics = {}
        
        for metric_name, batch_list in self.batch_metrics.items():
            if not batch_list:
                continue
                
            all_metrics = torch.cat(batch_list, dim=0)
            sample_means = torch.mean(all_metrics, dim=1)
            overall_mean = torch.mean(sample_means).item()
            final_metrics[f'mean_{metric_name}'] = overall_mean
            
            if self.config.get('log_sample_metrics', False):
                for i, value in enumerate(sample_means):
                    final_metrics[f'sample_{i}_{metric_name}'] = value.item()

        if self.cls_accuracy_batches > 0:
            avg_cls_accuracy = self.cls_accuracy_sum / self.cls_accuracy_batches
            final_metrics['mean_cls_accuracy'] = avg_cls_accuracy
     
        for position_idx, metrics_dict in self.position_metrics.items():
            for metric_name, batch_tensors in metrics_dict.items():
                all_values = torch.cat(batch_tensors, dim=0)
                position_mean = torch.mean(all_values).item()
                final_metrics[f'position_{position_idx}_{metric_name}'] = position_mean
                
        return avg_loss, final_metrics
    
    def get_progress_info(self):
        """Get progress bar display information."""
        progress_info = {}
        
        for metric_name, batch_list in self.batch_metrics.items():
            if not batch_list or metric_name not in ['iou', 'dice']:
                continue
                
            all_metrics = torch.cat(batch_list, dim=0)
            overall_mean = torch.mean(torch.mean(all_metrics, dim=1)).item()
            progress_info[f'{metric_name}'] = f"{overall_mean:.4f}"
        
        if self.cls_accuracy_batches > 0:
            progress_info['cls_acc'] = f"{self.cls_accuracy_sum / self.cls_accuracy_batches:.4f}"
            
        return progress_info

class MetricLogger:
    """Metric logger."""
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.logger = logging.getLogger(__name__)
    
    def log_metrics(self, metrics, epoch, phase='val'):
        for metric in ['iou', 'dice', 'precision', 'recall', 'f1', 'cls_accuracy']:
            mean_key = f'mean_{metric}'
            
            if mean_key in metrics:
                self.writer.add_scalar(f'{phase}/{metric}', metrics[mean_key], epoch)
                
                self.logger.info(
                    f'{phase.capitalize()} Epoch {epoch} - {metric}: '
                    f'{metrics[mean_key]:.4f}'
                )