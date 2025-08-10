import torch
import numpy as np
from tqdm import tqdm
import logging
import json
from pathlib import Path
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from model.loss.lwcaLR import LinearWarmupCosineAnnealingLR
from model.loss.lossall_match import EndToEndTeethLoss
from model.dataset.hdf5 import convert_to_hdf5
from model.PEGnet import DentalSegmentationSystem
from model.dataset.hdf5dataset_aug_jaw import HDF5TeethDataset
from utils.Metric_match import MetricCalculator, MetricLogger

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.use_amp = config.get('use_amp', True)
        amp_dtype_cfg = config.get('amp_dtype', 'float16').lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_cfg == 'bfloat16' else torch.float16
        self.use_scaler = self.use_amp and self.amp_dtype == torch.float16
        if self.use_scaler:
            self.scaler = torch.amp.GradScaler()
            
        self.set_seed(config.get('seed', 42))
        self.setup_directories()
        self.logger = self.setup_logging()
        
        self.setup_data()
        self.setup_test_data()
        self.setup_model()
        
        self.metric_logger = MetricLogger(self.log_dir)
        self.metric_calculator = MetricCalculator(config)
        
    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
    def setup_directories(self):
        """Create necessary directories."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = Path(self.config['save_dir']) / timestamp
        self.log_dir = self.save_dir / 'logs'
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.tensorboard_dir = self.save_dir / 'tensorboard'
        
        for dir_path in [self.log_dir, self.checkpoint_dir, self.tensorboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.writer = SummaryWriter(self.tensorboard_dir)
        
    def setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'train.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger()
    
    def setup_data(self):
        """Initialize datasets and dataloaders."""
        h5_path = Path(self.config['h5_path'])
        
        if not h5_path.exists():
            self.logger.info(f"HDF5 file {h5_path} not found, starting conversion from NPZ files...")
            convert_to_hdf5(self.config['data_dir'], h5_path)
            self.logger.info(f"HDF5 file created: {h5_path}")

        view_indices = self.config.get('view_indices', None)
        train_dataset = HDF5TeethDataset(
            h5_path,
            transform=True,
            split_file_path=self.config['train_split_path'],
            view_indices=view_indices
        )
        val_dataset = HDF5TeethDataset(
            h5_path,
            split_file_path=self.config['val_split_path'],
            view_indices=view_indices
        )
        batch_size = self.config['batch_size']
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )
            
    def setup_test_data(self):
        """Initialize test dataset."""
        h5_path = Path(self.config['h5_path'])
        
        if not h5_path.exists():
            self.logger.error(f"Test HDF5 file {h5_path} not found. Please run training first to generate it.")
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
        view_indices = self.config.get('view_indices', None)
        test_dataset = HDF5TeethDataset(
            h5_path,
            split_file_path=self.config['test_split_path'],
            view_indices=view_indices
        )
        batch_size = self.config['batch_size']
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )
         
    def setup_model(self):
        """Initialize model, loss function, and optimizer."""

        self.model = DentalSegmentationSystem(self.config).to(self.device)
        
        # Freeze prompt generator parameters
        freeze_prompt_generator = self.config.get('freeze_prompt_generator', True)
        if freeze_prompt_generator:
            for param in self.model.auto_prompt_generator.parameters():
                param.requires_grad = False

        # Freeze classification head parameters
        freeze_classification_heads = self.config.get('freeze_classification_heads', False)
        if freeze_classification_heads:
            for param in self.model.class_head.parameters():
                param.requires_grad = False

         # Freeze SAM model parameters (will not be trained)
        if not self.config.get('finetune_sam', False):
            for param in self.model.sam_model.parameters():
                param.requires_grad = False
        
        param_groups = []
        
        prompt_params = list(self.model.auto_prompt_generator.parameters())
        if prompt_params and any(p.requires_grad for p in prompt_params):
            param_groups.append({
                'params': prompt_params, 
                'lr': self.config.get('prompt_lr', 1e-4)
            })
        if not freeze_classification_heads:
            classification_params = list(self.model.class_head.parameters())
            if classification_params and any(p.requires_grad for p in classification_params):
                param_groups.append({
                    'params': classification_params,
                    'lr': self.config.get('classification_lr', 2e-4)
                })
        if not self.config.get('freeze_refine_net', False):
            refine_params = list(self.model.refine_net.parameters())
            if refine_params and any(p.requires_grad for p in refine_params):
                param_groups.append({
                    'params': refine_params, 
                    'lr': self.config['learning_rate']
                })
        if self.config.get('finetune_sam', False):
            sam_params = list(self.model.sam_model.parameters())
            if sam_params and any(p.requires_grad for p in sam_params):
                param_groups.append({
                    'params': sam_params, 
                    'lr': self.config.get('sam_lr', 1e-5)
                })

        cls_loss_weight = self.config.get('cls_loss_weight', 1.0)
        cost_class = self.config.get('cost_class', 1.0)
        cost_mask = self.config.get('cost_mask', 5.0)
        cost_dice = self.config.get('cost_dice', 1.0)
        
        self.criterion = EndToEndTeethLoss(
            cls_loss_weight=cls_loss_weight,
            cost_class=cost_class,
            cost_mask=cost_mask,
            cost_dice=cost_dice
        ).to(self.device)
        
        uncertainty_params_apg = list(self.criterion.uncertainty_weighting_apg.parameters())
        if uncertainty_params_apg and any(p.requires_grad for p in uncertainty_params_apg):
            param_groups.append({
                'params': uncertainty_params_apg,
                'lr': self.config.get('uncertainty_lr', 1e-3) 
            })
        uncertainty_params_refine = list(self.criterion.uncertainty_weighting_refine.parameters())
        if uncertainty_params_refine and any(p.requires_grad for p in uncertainty_params_refine):
            param_groups.append({
                'params': uncertainty_params_refine,
                'lr': self.config.get('uncertainty_lr', 1e-3)
            })

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config['weight_decay']
        )
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=self.config['warmup_epochs'],
            max_epochs=self.config['epochs'],
            warmup_start_lr=1e-6,
            eta_min=1e-6,
        )

    def rearrange_predictions_by_match(self, refined_logits, indices, gt_masks_shape):
        """
        Rearranges prediction masks based on Hungarian matching results to align their channel order with gt_masks.
        """
        B, _, H, W = refined_logits.shape
        C = gt_masks_shape[1]
        device = refined_logits.device

        rearranged_masks = torch.zeros((B, C, H, W), device=device, dtype=refined_logits.dtype)
        rearranged_masks[:, 0] = refined_logits[:, 0]
        for b in range(B):
            pred_indices, gt_indices = indices[b]
            if len(pred_indices) == 0:
                continue
            refined_pred_channels = pred_indices + 1
            gt_target_channels = gt_indices + 1
            rearranged_masks[b, gt_target_channels] = refined_logits[b, refined_pred_channels]

        return rearranged_masks

    def rearrange_predictions_for_inference(self, refined_probs, class_logits):
        """
        For inference without GT, rearranges predictions by uniquely assigning the 16 query predictions to 16 class channels using linear assignment.
        """
        B, _, H, W = refined_probs.shape
        device = refined_probs.device
        dtype = refined_probs.dtype

        rearranged_masks_final = torch.zeros((B, 17, H, W), device=device, dtype=dtype)
        rearranged_masks_final[:, 0] = refined_probs[:, 0]
        pred_class_probs = F.softmax(class_logits, dim=-1).to(torch.float32)

        for b in range(B):
            cost_matrix_b = -pred_class_probs[b].cpu().numpy()
            query_indices_assigned, class_indices_assigned = linear_sum_assignment(cost_matrix_b)
            for i in range(len(query_indices_assigned)):
                query_idx = query_indices_assigned[i]
                assigned_class_idx = class_indices_assigned[i]
                mask_for_query = refined_probs[b, query_idx + 1]
                rearranged_masks_final[b, assigned_class_idx + 1] = mask_for_query
            
        return rearranged_masks_final
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'uncertainty_weighting_state_dict': self.criterion.state_dict(),
            'config': self.config
        }
        if self.use_scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        self.logger.info(f'Checkpoint saved: {checkpoint_path}')
        
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model weights, allowing for non-strict matching
        self.logger.info(f"Loading model weights from checkpoint {checkpoint_path}...")
        load_result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
        # Only attempt to load optimizer and scheduler states if model weights loaded perfectly
        if not load_result.missing_keys and not load_result.unexpected_keys:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # self.logger.info("Optimizer and scheduler states loaded from checkpoint.")
            except ValueError as e:
                self.logger.warning(f"Failed to load optimizer/scheduler state: {e}. Using new optimizer/scheduler.")

        if 'uncertainty_weighting_state_dict' in checkpoint:
            self.criterion.load_state_dict(checkpoint['uncertainty_weighting_state_dict'], strict=False)
    
        # Load scaler state (if it exists and is needed)
        if self.use_scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['epoch']

    def train(self):
        """Train the model (can resume from a checkpoint).""" 
        start_epoch = 1
        if self.config.get('resume_checkpoint') and Path(self.config['resume_checkpoint']).exists():
            self.logger.info(f"Resuming from checkpoint: {self.config['resume_checkpoint']}")
            start_epoch = self.load_checkpoint(self.config['resume_checkpoint']) + 1
            self.logger.info(f"Training will continue from epoch {start_epoch}")

        best_val_loss = float('inf')
        best_val_iou = 0
        best_cls_accuracy = 0  # Track classification accuracy
        # Set validation interval
        val_interval = self.config.get('val_interval')  # Validation frequency

        for epoch in range(start_epoch, self.config['epochs'] + 1):
            # Train for one epoch
            train_loss, train_metrics = self.train_epoch(epoch)
        
            # Condition for running validation: every val_interval epochs or the last epoch
            do_validation = (epoch % val_interval == 0) or (epoch == self.config['epochs'])
            
            save = False
            if do_validation:
                # Validation
                val_loss, val_metrics = self.validate(epoch)

                # Save the best model based on multiple metrics
                if val_metrics.get('mean_iou', 0) > best_val_iou and not save:
                    best_val_iou = val_metrics['mean_iou']
                    self.save_checkpoint(epoch, is_best=True)
                    save = True
                    self.logger.info(f"Epoch {epoch}: New best mIoU: {best_val_iou:.4f}")
                
                if val_loss < best_val_loss and not save:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    save = True
                    self.logger.info(f"Epoch {epoch}: New best validation loss: {best_val_loss:.4f}")
                
            # Save periodically
            if epoch % self.config['save_freq'] == 0 and not save:
                self.save_checkpoint(epoch)
                save = True

            # Adjust learning rate (needed every epoch)
            self.scheduler.step()
        
        self.metric_logger.writer.close()
           
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        # Reset metric calculator
        self.metric_calculator.reset()
        with tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Prepare data
                images = batch['image'].to(self.device)
                gt_masks = batch['gt_masks'].to(self.device)
                valid_masks = batch['valid_masks'].to(self.device)
                tooth_ids = batch['tooth_ids']  # Get tooth_ids for metric calculation
                
                # Get ground truth tooth class labels for classification loss
                gt_tooth_classes = self.prepare_tooth_classes(batch).to(self.device)
                
                # Prepare gt_data
                gt_data = {
                    'gt_masks': gt_masks,
                    'valid_masks': valid_masks,
                    'gt_tooth_classes': gt_tooth_classes
                }
                
                # Forward pass
                self.optimizer.zero_grad()
                
                # Use mixed precision
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                        # End-to-end forward pass
                        sam_masks, refined_masks, labels, class_logits = self.model(images)
                        
                        # Prepare stage1_outputs and stage2_outputs
                        stage1_outputs = {
                            'sam_masks': sam_masks,
                            'confidence': labels,
                            'class_logits': class_logits
                        }
                        stage2_outputs = {
                            'refined_logits': refined_masks
                        }
                        
                        # Calculate loss
                        loss, losses_dict, match_indices = self.criterion(stage1_outputs, stage2_outputs, gt_data)
                        
                    # Backward pass and parameter update
                    if self.use_scaler:
                        self.scaler.scale(loss).backward()

                        if self.config.get('grad_clip', 0) > 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config['grad_clip']
                            )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()

                        if self.config.get('grad_clip', 0) > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config['grad_clip']
                            )
                        self.optimizer.step()
                else:
                    # Original full-precision computation
                    sam_masks, refined_masks, labels, class_logits = self.model(images)
                    
                    # Prepare stage1_outputs and stage2_outputs
                    stage1_outputs = {
                        'sam_masks': sam_masks,
                        'confidence': labels,
                        'class_logits': class_logits
                    }
                    stage2_outputs = {
                        'refined_logits': refined_masks
                    }
                    
                    loss, losses_dict, match_indices = self.criterion(stage1_outputs, stage2_outputs, gt_data)
                    loss.backward()
                    
                    if self.config.get('grad_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['grad_clip']
                        )
                        
                    self.optimizer.step()
                
                # Rearrange predictions for metric calculation
                refined_probs = F.softmax(refined_masks, dim=1)
                rearranged_probs = self.rearrange_predictions_by_match(
                    refined_probs, match_indices, gt_masks.shape
                )
                
                # Update metric calculator
                self.metric_calculator.update(rearranged_probs, gt_masks, tooth_ids, loss, losses_dict['cls_accuracy'])
                
                # Get progress info
                progress_info = self.metric_calculator.get_progress_info()
                
                # Add classification loss to progress info
                if 'cls_loss' in losses_dict:
                    progress_info['cls_loss'] = f"{losses_dict['cls_loss']:.4f}"
            
                # Update progress bar to show loss and metrics
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    **progress_info
                })
                    
                # Log the training process
                if batch_idx % self.config['log_freq'] == 0:
                    step = epoch * len(self.train_loader) + batch_idx
                    self.log_training_step(
                        step, loss.item(), losses_dict
                    )
                        
        # Calculate epoch average metrics
        avg_loss, final_metrics = self.metric_calculator.compute()
        
        # Log metrics to tensorboard
        self.metric_logger.log_metrics(final_metrics, epoch, phase='train')
        
        return avg_loss, final_metrics
    
    def prepare_tooth_classes(self, batch):
        """
        Prepare tooth class labels from the batch.
        
        Args:
            batch: Data batch.
            
        Returns:
            gt_tooth_classes: Tensor of shape [B, 16] representing the ground truth class (0-15) for each tooth.
        """
        # Assuming tooth_ids are 1-16 tooth numbers
        tooth_ids = batch['tooth_ids'].clone()  # Avoid modifying original data
        
        # Convert from 1-16 to 0-15 class indices
        gt_tooth_classes = tooth_ids - 1
        
        # Ensure values are within the valid range (0-15)
        gt_tooth_classes = torch.clamp(gt_tooth_classes, 0, 15)
        
        return gt_tooth_classes
    
    @torch.no_grad()
    def validate(self, epoch):
        """Validation function."""
        self.model.eval()
        
        val_loss = 0
        # Reset metric calculator
        self.metric_calculator.reset()
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Prepare data
                    images = batch['image'].to(self.device)
                    gt_masks = batch['gt_masks'].to(self.device)
                    tooth_ids = batch['tooth_ids']
                    valid_masks = batch['valid_masks'].to(self.device)
                    
                    # Get ground truth tooth class labels
                    gt_tooth_classes = self.prepare_tooth_classes(batch).to(self.device)
                    
                    gt_data = {
                        'gt_masks': gt_masks,
                        'valid_masks': valid_masks,
                        'gt_tooth_classes': gt_tooth_classes
                    }
                    
                    # Use mixed precision
                    if self.use_amp:
                        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                            # New model output format
                            sam_masks, refined_masks, labels, class_logits = self.model(images)
                            
                            stage1_outputs = {
                                'sam_masks': sam_masks,
                                'confidence': labels,
                                'class_logits': class_logits
                            }
                            stage2_outputs = {
                                'refined_logits': refined_masks
                            }
                            
                            # Calculate loss
                            loss, losses_dict, _ = self.criterion(stage1_outputs, stage2_outputs, gt_data)
                    else:
                        # Original full-precision computation
                        sam_masks, refined_masks, labels, class_logits = self.model(images)
                        
                        stage1_outputs = {
                            'sam_masks': sam_masks,
                            'confidence': labels,
                            'class_logits': class_logits,
                        }
                        stage2_outputs = {
                            'refined_logits': refined_masks
                        }
                        
                        # Calculate loss
                        loss, losses_dict, _ = self.criterion(stage1_outputs, stage2_outputs, gt_data)
                    
                    # Rearrange predictions (using new method without GT)
                    refined_probs = F.softmax(refined_masks, dim=1)
                    rearranged_probs = self.rearrange_predictions_for_inference(
                        refined_probs, class_logits
                    )
                    
                    # Update metric calculator (can still calculate some prediction stats without GT)
                    self.metric_calculator.update(rearranged_probs, gt_masks, tooth_ids, loss, losses_dict['cls_accuracy'])
                    
                    # Accumulate loss
                    val_loss += loss.item()
                    
                    # Get current progress info
                    progress_info = self.metric_calculator.get_progress_info()
                    
                    # Add classification loss to progress info
                    if 'cls_loss' in losses_dict:
                        progress_info['cls_loss'] = f"{losses_dict['cls_loss']:.4f}"
                    
                    # Show some metrics for the current batch
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        **progress_info
                    })
                    
                    # Log validation images
                    if batch_idx % self.config['log_freq'] == 0:
                        step = epoch * len(self.val_loader) + batch_idx
                        self.log_validation_step(
                            step, loss.item(), losses_dict
                        )
            
            # Calculate final metrics
            avg_loss, final_metrics = self.metric_calculator.compute()
            
            # Log metrics to tensorboard
            self.metric_logger.log_metrics(final_metrics, epoch, phase='val')

            return avg_loss, final_metrics
            

    @torch.no_grad()
    def test(self, checkpoint_path=None):
        """
        Evaluate the model on the test set and save predictions (saved in class order).
        
        Args:
            checkpoint_path: Path to the checkpoint to load, if None, use the current model.
        """
        # Load checkpoint (if specified)
        if checkpoint_path is not None:
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Ensure test data is loaded
        if not hasattr(self, 'test_loader'):
            self.setup_test_data()
        
        # Create test results directory
        test_dir = self.save_dir / 'test_results'
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize and reset metric calculator
        self.metric_calculator.reset()
        
        # Organize predictions by case
        case_predictions = {}
        
        with torch.no_grad():
            with tqdm(self.test_loader, desc='test') as pbar:
                for batch_idx, batch in enumerate(pbar):
                    # Prepare data
                    images = batch['image'].to(self.device)
                    gt_masks = batch['gt_masks'].to(self.device)
                    tooth_ids = batch['tooth_ids']
                    case_names = batch['case_name']
                    view_idxs = batch['view_idx']
                    valid_masks = batch['valid_masks'].to(self.device)
                    
                    # Get ground truth tooth class labels
                    gt_tooth_classes = self.prepare_tooth_classes(batch).to(self.device)
                    
                    gt_data = {
                        'gt_masks': gt_masks,
                        'valid_masks': valid_masks,
                        'gt_tooth_classes': gt_tooth_classes
                    }
                    
                    if self.use_amp:
                        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                            # Forward pass
                            sam_masks, refined_masks, labels, class_logits = self.model(images)
                            
                            stage1_outputs = {
                                'sam_masks': sam_masks,
                                'confidence': labels,
                                'class_logits': class_logits,
                            }
                            stage2_outputs = {
                                'refined_logits': refined_masks
                            }
                            loss, losses_dict, _ = self.criterion(stage1_outputs, stage2_outputs, gt_data)
                    else:
                        # Full-precision computation
                        sam_masks, refined_masks, labels, class_logits = self.model(images)
                        
                        stage1_outputs = {
                            'sam_masks': sam_masks,
                            'confidence': labels,
                            'class_logits': class_logits,
                        }
                        stage2_outputs = {
                            'refined_logits': refined_masks
                        }
                        loss, losses_dict, _ = self.criterion(stage1_outputs, stage2_outputs, gt_data)

                    # Rearrange predictions
                    refined_probs = F.softmax(refined_masks, dim=1)
                    rearranged_probs = self.rearrange_predictions_for_inference(
                        refined_probs, class_logits
                    )
                    
                    # Update metric calculator
                    self.metric_calculator.update(rearranged_probs, gt_masks, tooth_ids, loss, losses_dict['cls_accuracy'])
                    
                    # Get progress info
                    progress_info = self.metric_calculator.get_progress_info()
                    
                    # Add classification metrics
                    if 'cls_loss' in losses_dict:
                        progress_info['cls_loss'] = f"{losses_dict['cls_loss']:.4f}"
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        **progress_info
                    })
                    
                    # Get classification predictions
                    class_probs = F.softmax(class_logits, dim=-1)  # [B, 16, 16]
                    
                    # Process each sample in the batch, save in class order
                    for i in range(len(images)):
                        case_name = case_names[i]
                        view_idx = view_idxs[i].item()
                        curr_tooth_ids = tooth_ids[i]

                        num_views = 4
                        if case_name not in case_predictions:
                            case_predictions[case_name] = {
                                'masks': np.zeros((num_views, 16, 512, 512), dtype=bool),
                                'tooth_ids': np.zeros((num_views, 16), dtype=np.int32),
                                'valid_masks': np.zeros((num_views, 16), dtype=bool)
                            }
                        
                        # Get current case data
                        case_data = case_predictions[case_name]
                        curr_rearranged_probs = rearranged_probs[i]  # [17, H, W]
                        
                        # Extract tooth channels from predicted masks (skip background channel)
                        for j, tooth_id in enumerate(curr_tooth_ids):
                            tooth_id = int(tooth_id)
                            if tooth_id <= 16 and tooth_id >= 1:  # Ensure tooth_id is in the 1-16 range
                                # Get the corresponding mask from prediction
                                array_idx = tooth_id  # Index in model output, usually channel 0 is background, so tooth 1 corresponds to index 1
                                mask = (curr_rearranged_probs[array_idx] > 0.5).cpu().numpy()  
                                
                                # Convert to 0-indexed for saving to result array
                                array_pos = tooth_id - 1  # Convert 1-16 tooth ID to 0-15 array index
                                case_data['masks'][view_idx, array_pos] = mask
                                case_data['tooth_ids'][view_idx, array_pos] = tooth_id  # Save original tooth_id (1-16)
                                case_data['valid_masks'][view_idx, array_pos] = True

        # Save predictions for each case
        pred_dir = test_dir / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        for case_name, case_data in case_predictions.items():
            output_file = pred_dir / f"{case_name}_pred_masks.npz"
            np.savez(
                output_file,
                masks=case_data['masks'],  # [7, 16, H, W] - saved in class order
                tooth_ids=case_data['tooth_ids'],  # [7, 16] - class indices 0-15
                valid_masks=case_data['valid_masks']  # [7, 16] - confidence scores
            )
        
        # Calculate final metrics
        avg_loss, final_metrics = self.metric_calculator.compute()
        
        # Save results to a JSON file
        results = {
            'loss': avg_loss,
            'metrics': final_metrics,
            'checkpoint': str(checkpoint_path) if checkpoint_path else 'current_model'
        }
        
        with open(test_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        self.logger.info(f"Test results saved to {test_dir}")
        
        return avg_loss, final_metrics, pred_dir
    
    def log_training_step(self, step, loss, losses_dict):
        """Log detailed information for a training step."""
        # Log loss
        self.writer.add_scalar('Train/Loss', loss, step)
        for k, v in losses_dict.items():
            if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1):
                self.writer.add_scalar(f'Train/{k}', v, step)
        
        # Log classification loss and accuracy
        if 'cls_loss' in losses_dict:
            self.writer.add_scalar('Train/cls_loss', losses_dict['cls_loss'], step)
        if 'cls_accuracy' in losses_dict:
            self.writer.add_scalar('Train/cls_accuracy', losses_dict['cls_accuracy'], step)
        
        # Log learning rate
        self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], step)
        
    def log_validation_step(self, step, loss, losses_dict):
        """Log visualization results for a validation step."""
        # Log loss
        self.writer.add_scalar('Val/Loss', loss, step)
        for k, v in losses_dict.items():
            if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1):
                self.writer.add_scalar(f'Val/{k}', v, step)
        
        # Log classification loss and accuracy
        if 'cls_loss' in losses_dict:
            self.writer.add_scalar('Val/cls_loss', losses_dict['cls_loss'], step)
        if 'cls_accuracy' in losses_dict:
            self.writer.add_scalar('Val/cls_accuracy', losses_dict['cls_accuracy'], step)
