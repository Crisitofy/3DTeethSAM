import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import numpy as np
import random

class TeethAugmentation:
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        # Brightness and contrast adjustment
        self.brightness_prob = config.get('brightness_prob', 0.5)
        self.brightness_range = config.get('brightness_range', 0.15)
        
        self.contrast_prob = config.get('contrast_prob', 0.5)
        self.contrast_range = config.get('contrast_range', 0.15)
        
        # Add noise
        self.noise_prob = config.get('noise_prob', 0.3)
        self.noise_std = config.get('noise_std', 0.01)
        
        # Small rotation (optional)
        self.rotate_prob = config.get('rotate_prob', 0.0)  # Disabled by default
        self.rotate_range = config.get('rotate_range', 5)  # ±5 degrees
    
    def __call__(self, image, gt_masks):
        """
        Apply data augmentation.
        Args:
            image: [3, H, W] image tensor.
            gt_masks: [17, H, W] mask tensor (including background).
        Returns:
            augmented_image, augmented_masks
        """
        # Clone inputs to avoid modifying original data
        image = image.clone()
        gt_masks = gt_masks.clone()
        
        # 1. Brightness adjustment
        if random.random() < self.brightness_prob:
            brightness_factor = 1 + random.uniform(-self.brightness_range, self.brightness_range)
            image = image * brightness_factor
        
        # 2. Contrast adjustment
        if random.random() < self.contrast_prob:
            contrast_factor = 1 + random.uniform(-self.contrast_range, self.contrast_range)
            mean = image.mean(dim=[1, 2], keepdim=True)
            image = (image - mean) * contrast_factor + mean
        
        # 3. Add Gaussian noise
        if random.random() < self.noise_prob:
            noise = torch.randn_like(image) * self.noise_std
            image = image + noise
        
        # 4. Small rotation (applied to both image and masks)
        if self.rotate_prob > 0 and random.random() < self.rotate_prob:
            angle = random.uniform(-self.rotate_range, self.rotate_range)
            image, gt_masks = self._rotate(image, gt_masks, angle)
        
        # Ensure values are within a valid range
        image = torch.clamp(image, 0, 1)
        gt_masks = torch.clamp(gt_masks, 0, 1)
        
        return image, gt_masks
    
    def _rotate(self, image, masks, angle):
        """Rotation implementation using PyTorch."""
        import torch.nn.functional as F
        
        # Convert angle to radians
        theta = angle * np.pi / 180
        
        # Create rotation matrix
        cos_val = np.cos(theta)
        sin_val = np.sin(theta)
        
        # 2x3 affine transformation matrix
        rotation_matrix = torch.tensor([
            [cos_val, -sin_val, 0],
            [sin_val, cos_val, 0]
        ], dtype=torch.float32)
        
        # Add batch dimension
        image_batch = image.unsqueeze(0)
        masks_batch = masks.unsqueeze(0)
        
        # Create grid
        grid = F.affine_grid(
            rotation_matrix.unsqueeze(0),
            image_batch.size(),
            align_corners=False
        )
        
        # Apply rotation
        rotated_image = F.grid_sample(
            image_batch, grid,
            mode='bilinear',
            padding_mode='reflection',
            align_corners=False
        )
        
        rotated_masks = F.grid_sample(
            masks_batch, grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=False
        )
        
        return rotated_image.squeeze(0), rotated_masks.squeeze(0)

class HDF5TeethDataset(Dataset):
    """HDF5 dataset for tooth segmentation."""
    def __init__(self, hdf5_file, transform=False, mode=None, augment_config=None, jaw_type=None, split_file_path=None, view_indices=None):
        """
        Args:
            hdf5_file: Path to the HDF5 file.
            transform: Whether to apply data augmentation.
            mode: 'train', 'val', 'test', 'test_all'.
            augment_config: Dictionary of augmentation configurations.
            split_file_path: Path to the dataset split file.
        """
        self.transform = transform
        self.file_path = hdf5_file
        self.mode = mode
        self.jaw_type = jaw_type

        # View filtering list, None means keep all views (for ablation)
        if view_indices is not None:
            # Supports comma-separated string or iterable
            if isinstance(view_indices, str):
                view_indices = [int(v) for v in view_indices.split(',') if v.strip()]
            self.view_indices = set(view_indices)
        else:
            self.view_indices = None

        # Open HDF5 file (read-only mode)
        self.h5_file = h5py.File(hdf5_file, 'r')

        # Load all sample indices
        all_samples = json.loads(self.h5_file['sample_index'][()])
        
        # Filter samples based on the dataset split file
        if split_file_path:
            with open(split_file_path, 'r') as f:
                split_case_names = {line.strip() for line in f if line.strip()}
            
            self.samples = []
            for sample in all_samples:
                # Assuming case_name format is P001_upper or P001_lower
                case_name = sample['case_name'].split('_')[0]+'_'+sample['case_name'].split('_')[1]
                if case_name in split_case_names:
                    self.samples.append(sample)
            print(f"Loaded {len(self.samples)} samples according to {split_file_path}")
        else:
            self.samples = all_samples
            print(f"Warning: No split file provided, loaded all {len(self.samples)} samples")

        # Filter samples by jaw_type (if needed)
        if jaw_type is not None:
            jaw_samples = [s for s in self.samples if f"_{jaw_type}_" in s['case_name']]
            print(f"Filtered {len(jaw_samples)} {jaw_type} jaw samples from the split")
            self.samples = jaw_samples

        # Filter samples by view indices (if needed)
        if self.view_indices is not None:
            view_filtered_samples = [s for s in self.samples if s['view_idx'] in self.view_indices]
            print(f"Filtered to {len(view_filtered_samples)} samples based on view indices {sorted(self.view_indices)}")
            self.samples = view_filtered_samples
        
        # Check available fields in the dataset
        self.available_fields = list(self.h5_file.keys())
        required_fields = ['images', 'gt_masks']
            
        # Check if all required fields are available
        missing_fields = [field for field in required_fields if field not in self.available_fields]
        if missing_fields:
            raise ValueError(f"HDF5 file is missing required fields: {missing_fields}")
        
        # Apply augmentation only in training mode
        if mode == 'train' and transform:
            self.augmentation = TeethAugmentation(augment_config)
        else:
            self.augmentation = None
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['id']
        
        # Get basic data
        image = np.array(self.h5_file['images'][sample_id])
        gt_masks = np.array(self.h5_file['gt_masks'][sample_id])
        
        # Get metadata
        valid_masks = np.array(self.h5_file['metadata'][f"{sample_id}_valid_masks"])
        tooth_ids = np.array(self.h5_file['metadata'][f"{sample_id}_tooth_ids"])
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)  # [3, H, W]
        gt_masks_tensor = torch.from_numpy(gt_masks).float()  # [17, H, W]
        valid_masks_tensor = torch.from_numpy(valid_masks).long()  # [max_masks]
        
        # Apply data augmentation
        if self.augmentation is not None:
            image_tensor, gt_masks_tensor = self.augmentation(image_tensor, gt_masks_tensor)
        
        # Build the basic result dictionary
        result = {
            'case_name': sample['case_name'],
            'image': image_tensor,
            'gt_masks': gt_masks_tensor,
            'tooth_ids': tooth_ids,
            'view_idx': sample['view_idx'],
            'valid_masks': valid_masks_tensor
        }
            
        return result
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()