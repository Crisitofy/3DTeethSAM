import os
import argparse
import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import glob
import json
from scipy.optimize import linear_sum_assignment
import gc

from model.PEGnet import DentalSegmentationSystem
from preprocess.preprocess import ToothPreprocessor
from lift.lift3d import Projector, Labeler

import torch.nn.functional as F
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

UPPER_TEETH_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_TEETH_FDI = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
FDI_TO_LABEL = {}
for i, (upper, lower) in enumerate(zip(UPPER_TEETH_FDI, LOWER_TEETH_FDI)):
    FDI_TO_LABEL[upper] = i + 1 
    FDI_TO_LABEL[lower] = i + 1 

def load_labels(label_path):
    original_labels = np.loadtxt(label_path, dtype=np.int64)
    sem_labels = np.zeros_like(original_labels)
    for i, fdi_number in enumerate(original_labels):
        if fdi_number > 0 and fdi_number in FDI_TO_LABEL:
            sem_labels[i] = FDI_TO_LABEL[fdi_number]
        else:
            sem_labels[i] = 0
    return sem_labels, None

def load_labels_json(label_path):
    with open(label_path, 'r') as f:
        gt_data = json.load(f)
    original_labels = np.array(gt_data['labels'])
    sem_labels = np.zeros_like(original_labels)
    for i, fdi_number in enumerate(original_labels):
        if fdi_number > 0 and fdi_number in FDI_TO_LABEL:
            sem_labels[i] = FDI_TO_LABEL[fdi_number]
        else:
            sem_labels[i] = 0 
            
    return sem_labels, None

def setup_logger(log_dir, name='inference'):
    os.makedirs(log_dir, exist_ok=True)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{time_stamp}.log")
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers = []
    
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def generate_views(num_views): 
    """
    Generate a specified number of camera views (supports basic + anterior teeth orbit).
    """
    if num_views == 7: # Default 7 views
        return [
            [10, 0], [10, 45], [10, 315], [320, 0],
            [315, 45], [315, 315], [60, 0],
        ]
    
    views = []
    if num_views > 1:
        # Group 1: elevation change, covering front and back (from top to bottom)
        num_group1 = num_views // 3
        elevations = np.linspace(-80, 80, num_group1)
        for elev in elevations:
            views.append([round(elev), 0])
        # Group 2: azimuth change, covering left and right (horizontal rotation)
        num_group2 = num_views // 3
        azimuths = np.linspace(-80, 80, num_group2)
        for azim in azimuths:
            if abs(azim) > 1e-6:
                views.append([0, round(azim)])
        # New Group 3: oblique views orbiting the anterior teeth (oblique top-down)
        num_group3 = num_views - len(views)
        ring_azims = np.linspace(-80, 80, num_group3)
        ring_elevation = -45
        for azim in ring_azims:
            views.append([ring_elevation, round(azim)])
    return views

class InferencePipeline:
    def __init__(self, checkpoint_path, device='cuda', accelerate=False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.logger = setup_logger('logs/inference_logs')
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Acceleration enabled: {accelerate}")
        
        self.accelerate = accelerate
        self.config, self.model = self.load_model(checkpoint_path)
        
        self.labeler = Labeler(device=self.device)
        self.image_size = self.config.get('image_size', 512)

    def load_model(self, checkpoint_path):
        """Load model and configuration."""
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        config = checkpoint['config']
        
        # Adjust config parameters if needed
        config['finetune_sam'] = False # No finetuning during inference
        config['freeze_prompt_generator'] = True

        model = DentalSegmentationSystem(config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        self.logger.info("Model loaded successfully.")

        # Compile the model if acceleration is enabled
        if self.accelerate:
            self.logger.info("Applying torch.compile for acceleration. The first run will have a one-time compilation cost.")
            try:
                self.model = torch.compile(self.model)
                self.logger.info("torch.compile applied successfully.")
            except Exception as e:
                self.logger.warning(f"torch.compile failed with error: {e}. Proceeding without compilation.")

        return config, model

    def rearrange_predictions_for_inference(self, refined_probs, class_logits):
        B, _, H, W = refined_probs.shape
        device = refined_probs.device
        dtype = refined_probs.dtype

        rearranged_masks_final = torch.zeros((B, 17, H, W), device=device, dtype=dtype)
        rearranged_masks_final[:, 0] = refined_probs[:, 0]
        pred_class_probs = F.softmax(class_logits, dim=-1)

        for b in range(B):
            cost_matrix_b = -pred_class_probs[b].cpu().numpy()
            query_indices_assigned, class_indices_assigned = linear_sum_assignment(cost_matrix_b)
            for i in range(len(query_indices_assigned)):
                query_idx = query_indices_assigned[i]
                assigned_class_idx = class_indices_assigned[i]
                mask_for_query = refined_probs[b, query_idx + 1]
                rearranged_masks_final[b, assigned_class_idx + 1] = mask_for_query
            
        return rearranged_masks_final

    def preprocess(self, obj_path, views):
        
        self.logger.info(f"Preprocessing {obj_path} with {len(views)} views...")
        preprocessor = ToothPreprocessor(device=self.device)
        preprocessor.views = views 
        verts, faces = preprocessor.load_and_normalize_mesh(obj_path)
         
        num_original_verts = verts.shape[0]
        referenced_indices = torch.unique(faces.flatten())
        num_referenced_verts = referenced_indices.shape[0]
        if num_referenced_verts < num_original_verts:
            self.logger.info(f"Cleaning mesh: found {num_original_verts - num_referenced_verts} unreferenced vertices. Removing them.")
            verts = verts[referenced_indices]
            mapper = -torch.ones(num_original_verts, dtype=torch.long, device=self.device)
            mapper[referenced_indices] = torch.arange(num_referenced_verts, device=self.device)
            faces = mapper[faces]

        num_views = len(views)
        all_images = np.zeros((num_views, self.image_size, self.image_size, 3), dtype=np.float32)
        default_colors = torch.ones_like(verts, dtype=torch.float32) * 0.75
        _, light_directions = preprocessor.get_view_xyz(views)

        for i, (view, light_dir) in enumerate(zip(views, light_directions)):
            R, T = look_at_view_transform(
                dist=preprocessor.camera_distance,
                elev=view[0], azim=view[1], device=self.device
            )
            img = preprocessor.render_single_view(verts, faces, default_colors, R, T, light_dir)
            all_images[i] = img.cpu().numpy()
            
        self.logger.info("Preprocessing finished.")
        return all_images, verts, faces

    def project_to_3d(self, masks, verts, faces, views, view_weights=None):
        """
        Project 2D segmentation masks back to the 3D model.
        """
        self.logger.info("Projecting 2D masks to 3D mesh...")
        
        num_views = masks.shape[0]
        tooth_ids = np.tile(np.arange(1, 17), (num_views, 1))
        valid_masks = np.any(masks, axis=(2, 3))
        projector = Projector(device=self.device, image_size=self.image_size)
        projector.views = views
        
        projector.cameras = []
        for view in views:
            R, T = look_at_view_transform(
                dist=projector.camera_distance, elev=view[0], azim=view[1], device=self.device
            )
            projector.cameras.append(FoVPerspectiveCameras(
                device=self.device, R=R, T=T, znear=0.01, zfar=100.0,
                aspect_ratio=1.0, fov=60.0
            ))
        projected_points, visibility, depths, _ = projector.project(verts.cpu().numpy(), faces.cpu().numpy())
        assignments = projector.assign_masks(projected_points, visibility, masks, valid_masks)
        
        point_labels = self.labeler.vote(
            assignments, torch.tensor(tooth_ids, device=self.device), visibility, depths,
            view_weights=view_weights
        )
        final_labels = self.labeler.complete_segmentation_pipeline(
            verts.cpu().numpy(), faces.cpu().numpy(), point_labels, self.logger
        )
        return final_labels

    def run(self, obj_path, output_dir, num_views=7, use_equal_view_weights=False, gt_label_path=None, labels_save_dir=None):
        """
        Execute the full inference pipeline.
        """
        self.logger.info(f"Starting inference for {obj_path}")
        
        # 1. Generate views
        views = generate_views(num_views)
        
        # 2. Preprocess
        images, verts, faces = self.preprocess(obj_path, views)

        # 3. Model inference
        self.logger.info("Running model inference...")
        images_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).to(self.device)
        
        max_views_per_batch = 20
        num_views_total = images_tensor.shape[0]
        all_refined_masks = []
        all_class_logits = []

        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.accelerate):
            if num_views_total > max_views_per_batch:
                num_batches = -(-num_views_total // max_views_per_batch)
                for i in range(0, num_views_total, max_views_per_batch):
                    batch_images = images_tensor[i:i + max_views_per_batch]
                    self.logger.info(f"  Processing view batch {i // max_views_per_batch + 1}/{num_batches} ({i + 1}-{min(i + max_views_per_batch, num_views_total)})...")
                    _, batch_refined_masks, _, batch_class_logits = self.model(batch_images)
                    all_refined_masks.append(batch_refined_masks)
                    all_class_logits.append(batch_class_logits)
                refined_masks = torch.cat(all_refined_masks, dim=0)
                class_logits = torch.cat(all_class_logits, dim=0)
            else:
                _, refined_masks, _, class_logits = self.model(images_tensor)
        
        if self.accelerate:
            refined_masks = refined_masks.float()
            class_logits = class_logits.float()

        probs = F.softmax(refined_masks, dim=1)
        rearranged_probs = self.rearrange_predictions_for_inference(probs, class_logits)
        pred_masks = (rearranged_probs > 0.5)
        masks_no_bg = pred_masks[:, 1:, :, :].cpu().numpy()
        self.logger.info("Model inference finished.")

        # 4. 2D to 3D projection
        view_weights = None
        if use_equal_view_weights:
            view_weights = [1.0] * num_views
        final_labels = self.project_to_3d(masks_no_bg, verts, faces, views, view_weights=view_weights)
        
        # 5. Save results
        self.logger.info(f"Saving results to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        case_name = Path(obj_path).stem
        jaw_type = "unknown"
        if "_lower" in case_name.lower():
            jaw_type = "lower"
        elif "_upper" in case_name.lower():
            jaw_type = "upper"
        
        # Save label file to a specific directory
        if labels_save_dir:
            final_label_dir = os.path.join(labels_save_dir, jaw_type)
            os.makedirs(final_label_dir, exist_ok=True)
            label_file = os.path.join(final_label_dir, f"{case_name}.txt")
        else:
            label_file = os.path.join(output_dir, f"{case_name}.txt")
            
        np.savetxt(label_file, final_labels, fmt='%d')
        self.logger.info(f"Saved vertex labels to {label_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="End-to-end inference for 3D tooth segmentation.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the root data directory (e.g., containing 'lower_sim', 'upper_sim').")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--output_dir', type=str, default='inference', help="Directory to save the results.")
    parser.add_argument('--num_views', type=int, default=60, help="Number of views to generate for rendering.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use for inference ('cuda:0', 'cuda:1', 'cpu').")
    parser.add_argument('--equal_weights', action='store_true', help="If set, use equal view weights (1.0 for all) for 3D projection.")
    parser.add_argument('--test_split_file', type=str, default='preprocess/split/official/testing_all.txt', help="Optional path to a text file listing case names to process (one per line).")
    parser.add_argument('--accelerate', action='store_true', help="If set, enable inference acceleration with torch.compile and AMP.")
    
    args = parser.parse_args()

        # Robust GPU selection: Set CUDA_VISIBLE_DEVICES to ensure the correct physical GPU is used.
    if 'cuda:' in args.device:
        try:
            gpu_id = args.device.split(':')[-1]
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            # After setting the environment variable, the device for PyTorch should be just 'cuda'
            # as it will now refer to the only GPU visible to the process.
            args.device = 'cuda'
            print(f"INFO: CUDA_VISIBLE_DEVICES set to '{gpu_id}'. Physical GPU {gpu_id} will be used.")
        except (IndexError, ValueError):
            print(f"Warning: Could not parse GPU ID from device '{args.device}'. Defaulting to cuda:0.")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            args.device = 'cuda'
            
    input_dir = args.input_dir
    output_dir = args.output_dir

    labels_save_dir = os.path.join(output_dir, "labels")
    allowed_cases = None
    if args.test_split_file:
        if os.path.exists(args.test_split_file):
            with open(args.test_split_file, 'r') as f:
                allowed_cases = set(line.strip() for line in f if line.strip())
    
    obj_files_to_process = []
    for jaw in ['lower', 'upper']:
        jaw_data_dir = os.path.join(input_dir, f"{jaw}")
        
        if not os.path.isdir(jaw_data_dir):
            continue
        case_dirs = sorted([d for d in glob.glob(f"{jaw_data_dir}/*") if os.path.isdir(d)])
        print(f"Found {len(case_dirs)} potential case directories in '{jaw_data_dir}'.")

        for case_dir in case_dirs:
            case = os.path.basename(case_dir)
            if allowed_cases is not None:
                case_id = f"{case}_{jaw}"
                if case_id not in allowed_cases:
                    continue
            obj_path = os.path.join(case_dir, f"{case}_{jaw}.obj")
            if os.path.exists(obj_path):
                obj_files_to_process.append(obj_path)

    if not obj_files_to_process:
        print(f"No .obj files found in '{input_dir}' with the expected structure.")
    else:
        pipeline = InferencePipeline(
            checkpoint_path=args.checkpoint, 
            device=args.device,
            accelerate=args.accelerate
        )
        pipeline.logger.info(f"Found {len(obj_files_to_process)} .obj files to process.")

        for obj_path in sorted(obj_files_to_process):
            relative_path = os.path.relpath(os.path.dirname(obj_path), input_dir)
            if relative_path == ".":
                current_output_dir = output_dir
            else:
                current_output_dir = os.path.join(output_dir, relative_path)
            base_name = Path(obj_path).stem
            gt_label_path = os.path.join(os.path.dirname(obj_path), f"{base_name}.json")

            pipeline.logger.info("-" * 60)
            pipeline.logger.info(f"Processing: {obj_path}")
            pipeline.logger.info(f"Results will be saved to: {current_output_dir}")
            
            pipeline.run(
                obj_path=obj_path,
                output_dir=current_output_dir,
                num_views=args.num_views,
                use_equal_view_weights=args.equal_weights,
                gt_label_path=gt_label_path,
                labels_save_dir=labels_save_dir
            )

            # Clean PyTorch's VRAM cache to prepare for the next case
            if 'cuda' in args.device:
                torch.cuda.empty_cache()
                gc.collect()
            
        pipeline.logger.info("=" * 60)
        pipeline.logger.info("Batch processing finished for all files.") 
