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

from model.loss.lwcaLR import LinearWarmupCosineAnnealingLR
from model.loss.lossall import EndToEndTeethLoss
from model.dataset.hdf5 import convert_to_hdf5
from model.PEG import DentalSegmentationSystem
from model.dataset.hdf5dataset_aug_jaw import HDF5TeethDataset
from utils.Metric import MetricCalculator, MetricLogger

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
        
        # AMP training
        self.use_amp = config.get('use_amp', True)
        # Select AMP dtype: float16 or bfloat16
        amp_dtype_cfg = config.get('amp_dtype', 'float16').lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype_cfg == 'bfloat16' else torch.float16

        # GradScaler is only needed for float16; not for bfloat16
        self.use_scaler = self.use_amp and self.amp_dtype == torch.float16
        if self.use_scaler:
            self.scaler = torch.amp.GradScaler()
            
        # Set random seed
        self.set_seed(config.get('seed', 42))
        
        # Create directories for saving and tensorboard writer
        self.setup_directories()
        
        # Setup logging
        self.logger = self.setup_logging()
        
        # Initialize dataset and model
        self.setup_data()
        self.setup_test_data()
        self.setup_model()
        
        self.metric_logger = MetricLogger(self.log_dir)
        # Create metric calculator
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
        
        # Batch size
        batch_size = self.config['batch_size']

        # Set up DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            persistent_workers=True
        )
            
    def setup_test_data(self):
        """Initialize test dataset."""
        h5_path = Path(self.config['h5_path'])
        
        # Ensure HDF5 file exists
        if not h5_path.exists():
            self.logger.error(f"Test HDF5 file {h5_path} not found. Please run training first to generate it.")
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        # Use HDF5 dataset and pass the test set split file
        view_indices = self.config.get('view_indices', None)

        test_dataset = HDF5TeethDataset(
            h5_path,
            split_file_path=self.config['test_split_path'],
            view_indices=view_indices
        )
        
        # Choose batch size based on training mode
        batch_size = self.config['batch_size']
        
        # Set up DataLoader
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
        
        # Create the end-to-end model
        self.model = DentalSegmentationSystem(self.config).to(self.device)
        
        freeze_prompt_generator = self.config.get('freeze_prompt_generator', True)
        if freeze_prompt_generator:
            for param in self.model.auto_prompt_generator.parameters():
                param.requires_grad = False

        if not self.config.get('finetune_sam', False):
            for param in self.model.sam_model.parameters():
                param.requires_grad = False

        dsp_params = []
        other_params = []
        for name, param in self.model.sam_model.image_encoder.trunk.named_parameters():
            if 'dsp' in name:  # Parameters of DSP module
                param.requires_grad = True
                dsp_params.append(param)
            else:  # Other parameters
                param.requires_grad = False
                other_params.append(param)
                
        param_groups = []
        if dsp_params:
            param_groups.append({
                'params': dsp_params,
                'lr': self.config.get('dsp_lr', 2e-4),  # DSP-specific learning rate
                'weight_decay': self.config.get('dsp_weight_decay', 0.01)  # DSP weight decay
            })
        
        # Prompt generator parameter group
        prompt_params = list(self.model.auto_prompt_generator.parameters())
        if prompt_params and any(p.requires_grad for p in prompt_params):
            param_groups.append({
                'params': prompt_params, 
                'lr': self.config.get('prompt_lr', 1e-4),
                'weight_decay': self.config.get('prompt_weight_decay', 0.01)
            })
        
        # RefineNet parameter group
        if not self.config.get('freeze_refine_net', True):
            refine_params = list(self.model.refine_net.parameters())
            if refine_params and any(p.requires_grad for p in refine_params):
                param_groups.append({
                    'params': refine_params, 
                    'lr': self.config['learning_rate'],
                    'weight_decay': self.config.get('weight_decay', 0.01)
                })

        if self.config.get('finetune_sam', False):
            sam_params = list(self.model.sam_model.parameters())
            if sam_params and any(p.requires_grad for p in sam_params):
                param_groups.append({
                    'params': sam_params, 
                    'lr': self.config.get('sam_lr', 1e-5),
                    'weight_decay': self.config.get('sam_weight_decay', 0.01)
                })
            
        self.criterion = EndToEndTeethLoss().to(self.device)
        
        optimizer_type = self.config.get('optimizer', 'AdamW')
        if optimizer_type == 'AdamW':
            self.optimizer = torch.optim.AdamW(param_groups)
        elif optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        warmup_epochs = self.config.get('warmup_epochs', 5)
        total_epochs = self.config['epochs']
        
        self.scheduler = LinearWarmupCosineAnnealingLR(
            self.optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=total_epochs
        )
    
        
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
        
        self.logger.info(f"Loading model weights from checkpoint {checkpoint_path}...")
        load_result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        if load_result.missing_keys:
            self.logger.warning(f"Missing keys when loading model: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            self.logger.warning(f"Unexpected keys when loading model: {load_result.unexpected_keys}")
        
        if not load_result.missing_keys and not load_result.unexpected_keys:
            self.logger.info("Model weights loaded, all keys matched.")

        if not load_result.missing_keys and not load_result.unexpected_keys:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info("Optimizer and scheduler states loaded from checkpoint.")
            except ValueError as e:
                self.logger.warning(f"Failed to load optimizer/scheduler state: {e}. Using new optimizer/scheduler.")
        else:
            self.logger.warning("Optimizer and scheduler will be re-initialized because the model architecture has changed (missing/unexpected keys detected).")

        if 'uncertainty_weighting_state_dict' in checkpoint:
            self.criterion.load_state_dict(checkpoint['uncertainty_weighting_state_dict'], strict=False)
    
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
        # start_epoch = 6
        best_val_loss = float('inf')
        best_val_iou = 0
        # Set validation interval
        val_interval = self.config.get('val_interval')  # Validation frequency

        for epoch in range(start_epoch, self.config['epochs'] + 1):
            # Train for one epoch
            self.train_epoch(epoch)

            do_validation = (epoch % val_interval == 0) or (epoch == self.config['epochs'])
            save = False
            if do_validation:
                val_loss, val_metrics = self.validate(epoch)

                if val_metrics.get('mean_iou', 0) > best_val_iou and save == False:
                    best_val_iou = val_metrics['mean_iou']
                    self.save_checkpoint(epoch, is_best=True)
                    save = True
                    self.logger.info(f"Epoch {epoch}: New best mIoU: {best_val_iou:.4f}")
                
                if val_loss < best_val_loss and save == False:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
                    save = True
                    self.logger.info(f"Epoch {epoch}: New best validation loss: {best_val_loss:.4f}")
                
            if epoch % self.config['save_freq'] == 0 and save == False:
                self.save_checkpoint(epoch)
                save = True
            self.scheduler.step()
        
        self.metric_logger.writer.close()
           
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0
        self.metric_calculator.reset()
        with tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["epochs"]}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                gt_masks = batch['gt_masks'].to(self.device)
                valid_masks = batch['valid_masks'].to(self.device)
                tooth_ids = batch['tooth_ids']  
                gt_data = {
                    'gt_masks': gt_masks,
                    'valid_masks': valid_masks
                }
                # Forward pass
                self.optimizer.zero_grad()
                
                # Use mixed precision
                if self.use_amp:
                    with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                        sam_masks, refined_masks, confidence = self.model(images)
                        
                        stage1_outputs = {
                            'sam_masks': sam_masks,
                            'confidence': confidence
                        }
                        stage2_outputs = {
                            'refined_logits': refined_masks
                        }
                        loss, losses_dict = self.criterion(stage1_outputs, stage2_outputs, gt_data)
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

                    sam_masks, refined_masks, confidence = self.model(images)

                    stage1_outputs = {
                        'sam_masks': sam_masks,
                        'confidence': confidence
                    }
                    stage2_outputs = {
                        'refined_logits': refined_masks
                    }
                    loss, losses_dict = self.criterion(stage1_outputs, stage2_outputs, gt_data)

                    loss.backward()
                    
                    if self.config.get('grad_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['grad_clip']
                        )
                        
                    self.optimizer.step()

                refined_masks = F.softmax(refined_masks, dim=1)
                self.metric_calculator.update(refined_masks, gt_masks, tooth_ids, loss)

                progress_info = self.metric_calculator.get_progress_info()

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    **progress_info
                })

                if batch_idx % self.config['log_freq'] == 0:
                    step = epoch * len(self.train_loader) + batch_idx
                    self.log_training_step(
                        step, loss.item(), losses_dict
                    )
        avg_loss, final_metrics = self.metric_calculator.compute()
        self.metric_logger.log_metrics(final_metrics, epoch, phase='train')
        
        return avg_loss
    
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
                    gt_data = {
                        'gt_masks': gt_masks,
                        'valid_masks': valid_masks
                    }
                    if self.use_amp:
                        with torch.amp.autocast(device_type='cuda', dtype=self.amp_dtype):
                            sam_masks, refined_masks, confidence = self.model(images)
                            stage1_outputs = {
                                'sam_masks': sam_masks,
                                'confidence': confidence
                            }
                            stage2_outputs = {
                                'refined_logits': refined_masks
                            }
                            loss, losses_dict = self.criterion(stage1_outputs, stage2_outputs, gt_data)
                    else:
                        sam_masks, refined_masks, confidence = self.model(images)
                        stage1_outputs = {
                            'sam_masks': sam_masks,
                            'confidence': confidence
                        }
                        stage2_outputs = {
                            'refined_logits': refined_masks
                        }
                        loss, losses_dict = self.criterion(stage1_outputs, stage2_outputs, gt_data)

                    refined_masks = F.softmax(refined_masks, dim=1)
                    self.metric_calculator.update(refined_masks, gt_masks, tooth_ids, loss)
                    val_loss += loss.item()
                    progress_info = self.metric_calculator.get_progress_info()
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        **progress_info
                    })
                    
                    if batch_idx % self.config['log_freq'] == 0:
                        step = epoch * len(self.val_loader) + batch_idx
                        self.writer.add_scalar('Val/Loss', loss.item(), step)
                        for k, v in losses_dict.items():
                            self.writer.add_scalar(f'Val/{k}_Loss', v, step)
            avg_loss, final_metrics = self.metric_calculator.compute()
            self.metric_logger.log_metrics(final_metrics, epoch, phase='val')

            return avg_loss, final_metrics
            
    @torch.no_grad()
    def test(self, checkpoint_path=None):
        """
        Evaluate the model on the test set and save predictions.
        """
        if checkpoint_path is not None:
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if not hasattr(self, 'test_loader'):
            self.setup_test_data()
        test_dir = self.save_dir / 'test_results'
        test_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()
        self.metric_calculator.reset()
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
                    gt_data = {
                        'gt_masks': gt_masks,
                        'valid_masks': valid_masks
                    }
                    sam_masks, refined_masks, confidence = self.model(images)
                    stage1_outputs = {
                        'sam_masks': sam_masks,
                        'confidence': confidence
                    }
                    stage2_outputs = {
                        'refined_logits': refined_masks
                    }
                    loss, losses_dict = self.criterion(stage1_outputs, stage2_outputs, gt_data)
                    refined_masks = F.softmax(refined_masks, dim=1)
                    self.metric_calculator.update(refined_masks, gt_masks, tooth_ids, loss)
                    progress_info = self.metric_calculator.get_progress_info()
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        **progress_info
                    })
                    for i in range(len(images)):
                        case_name = case_names[i]
                        view_idx = view_idxs[i].item()
                        curr_tooth_ids = tooth_ids[i]
                        if case_name not in case_predictions:
                            num_views = self.config.get('num_views', 7)
                            case_predictions[case_name] = {
                                'masks': np.zeros((num_views, 16, 512, 512), dtype=bool),
                                'tooth_ids': np.zeros((num_views, 16), dtype=np.int32),
                                'valid_masks': np.zeros((num_views, 16), dtype=bool)
                            }
                        case_data = case_predictions[case_name]
                        curr_pred_masks = refined_masks[i] 
                        
                        for j, tooth_id in enumerate(curr_tooth_ids):
                            tooth_id = int(tooth_id)
                            if tooth_id <= 16 and tooth_id >= 1:
                                array_idx = tooth_id  
                                mask = (curr_pred_masks[array_idx] > 0.5).cpu().numpy()  
                                
                                array_pos = tooth_id - 1  
                                case_data['masks'][view_idx, array_pos] = mask
                                case_data['tooth_ids'][view_idx, array_pos] = tooth_id 
                                case_data['valid_masks'][view_idx, array_pos] = True
        
        # Save predictions for each case
        pred_dir = test_dir / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        for case_name, case_data in case_predictions.items():
            output_file = pred_dir / f"{case_name}_pred_masks.npz"
            np.savez(
                output_file,
                masks=case_data['masks'],
                tooth_ids=case_data['tooth_ids'],
                valid_masks=case_data['valid_masks']
            )
        avg_loss, final_metrics = self.metric_calculator.compute()
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
            self.writer.add_scalar(f'Train/{k}_Loss', v, step)
        
        # Log learning rate
        self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], step)
