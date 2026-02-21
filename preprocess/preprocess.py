import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torch
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    DirectionalLights,
    TexturesVertex
)
from pytorch3d.io import load_obj
import json
from pathlib import Path
from tqdm import tqdm
import glob
import argparse

UPPER_TEETH_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28] 
LOWER_TEETH_FDI = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48] 
FDI_TO_LABEL = {}
for i, (upper, lower) in enumerate(zip(UPPER_TEETH_FDI, LOWER_TEETH_FDI)):
    FDI_TO_LABEL[upper] = i + 1  
    FDI_TO_LABEL[lower] = i + 1 

class ToothPreprocessor:
    def __init__(self, device='cuda'):
        self.device = device
        self.image_size = 512 
        self.camera_distance = 70
        self.background_color = (1, 1, 1)
        
        self.views = [
            [10, 0],  
            [10, 45],  
            [10, 315], 
            [320, 0],    
            [315, 45],  
            [315, 315], 
            [60, 0],  
            [270, 0],  
            [0, 90],
            [0, 270] 
        ]
        self.setup_renderer()

    def setup_renderer(self):
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=True,
        )

    def get_view_xyz(self, views):
        """Convert (azim, elev) to (x,y,z) for both camera position and light direction"""
        camera_positions = []
        light_directions = []
        for view in views:
            elev = view[0] * np.pi / 180.0
            azim = view[1] * np.pi / 180.0

            x = np.cos(elev) * np.sin(azim)
            y = np.sin(elev)
            z = np.cos(elev) * np.cos(azim)
            
            pos = np.array([x, y, z])
            camera_positions.append(pos)
            light_directions.append(pos)
        return np.stack(camera_positions, 0), np.stack(light_directions, 0)

    def load_and_normalize_mesh(self, mesh_path):
        verts, faces, aux = load_obj(mesh_path)
        faces = faces.verts_idx
        verts = verts.to(self.device)
        faces = faces.to(self.device)
        center = verts.mean(dim=0)
        verts = verts - center
        return verts, faces

    def load_labels(self, label_path, jaw_type):
        with open(label_path, 'r') as f:
            gt_data = json.load(f)

        labels = np.array(gt_data['labels'])
        instances = np.array(gt_data['instances'])
        sem_labels = np.zeros_like(labels)
        valid_teeth = []
        unique_fdi_numbers = np.unique(labels)
        unique_fdi_numbers = unique_fdi_numbers[unique_fdi_numbers > 0] 
        
        for fdi_number in unique_fdi_numbers:
            if int(fdi_number) in FDI_TO_LABEL:
                unified_label = FDI_TO_LABEL[int(fdi_number)]
                sem_labels[labels == fdi_number] = unified_label
                valid_teeth.append(unified_label)
        valid_teeth = np.unique(valid_teeth)
        
        return sem_labels, instances, valid_teeth

    def render_single_view(self, verts, faces, vertex_colors, view_R, view_T, light_direction):
        cameras = FoVPerspectiveCameras(
            device=self.device,
            R=view_R,
            T=view_T,
            znear=0.01,
            zfar=100.0,
            aspect_ratio=1.0,
            fov=60.0
        )
        current_lights = DirectionalLights(
            device=self.device,
            direction=torch.tensor(np.array([light_direction]), device=self.device),
            ambient_color=((0.48, 0.48, 0.48),),  
            diffuse_color=((0.65, 0.65, 0.65),),  
            specular_color=((0.6, 0.6, 0.6),)  
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=current_lights
            )
        )
        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex(vertex_colors.unsqueeze(0))
        )
        images = renderer(mesh)
        rendered_image = images[0, ..., :3].clamp(0, 1)
        
        return rendered_image
    
    def generate_gt_masks_f(self, verts, faces, sem_labels, cameras, valid_teeth, max_masks=16):
        """
        Generate ground truth masks, considering occlusion.
        """
        masks = np.zeros((max_masks, self.image_size, self.image_size), dtype=bool)
        tooth_ids = np.zeros(max_masks, dtype=np.int32)
        valid_masks = np.zeros(max_masks, dtype=bool)
        
        vertex_labels = torch.from_numpy(sem_labels).to(self.device)
        face_vert_labels = vertex_labels[faces] 
        face_labels, _ = torch.mode(face_vert_labels, dim=1)
        full_mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=TexturesVertex(torch.ones_like(verts).unsqueeze(0))
        )
        
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings)
        fragments = rasterizer(full_mesh)
        face_idxs = fragments.pix_to_face[0, :, :, 0]
        semantic_map = torch.zeros_like(face_idxs, dtype=torch.long)
        valid_mask = face_idxs >= 0
        semantic_map[valid_mask] = face_labels[face_idxs[valid_mask]]
        
        for idx, tooth_id in enumerate(valid_teeth):
            if idx >= max_masks:
                break
            tooth_mask = (semantic_map == tooth_id) & valid_mask
            masks[tooth_id-1] = tooth_mask.cpu().numpy()
            tooth_ids[tooth_id-1] = tooth_id
            valid_masks[tooth_id-1] = torch.any(tooth_mask).cpu().numpy()
        
        return masks, tooth_ids, valid_masks
    
    def process_single_tooth(self, obj_path, label_path, save_dir, jaw_type):
        """Process a single tooth model."""
        
        fname = Path(obj_path).stem
        os.makedirs(save_dir, exist_ok=True)

        verts, faces = self.load_and_normalize_mesh(obj_path)
        sem_labels, ins_labels, valid_teeth = self.load_labels(label_path, jaw_type)
        
        num_views = len(self.views)
        max_masks = 16
        all_images = np.zeros((num_views, self.image_size, self.image_size, 3), dtype=np.float32)
        all_gt_masks = np.zeros((num_views, max_masks, self.image_size, self.image_size), dtype=bool)
        all_tooth_ids = np.zeros((num_views, max_masks), dtype=np.int32)
        all_valid_masks = np.zeros((num_views, max_masks), dtype=bool)
        
        default_colors = torch.ones_like(verts, dtype=torch.float32) * 0.75
        
        _, light_directions = self.get_view_xyz(self.views)
        
        for i, (view, light_dir) in enumerate(zip(self.views, light_directions)):
            R, T = look_at_view_transform(
                dist=self.camera_distance,
                elev=view[0],
                azim=view[1],
                device=self.device
            )
            img = self.render_single_view(verts, faces, default_colors, R, T, light_dir)
            all_images[i] = img.cpu().numpy()
            cameras = FoVPerspectiveCameras(
                device=self.device,
                R=R,
                T=T,
                znear=0.01,
                zfar=100.0,
                aspect_ratio=1.0,
                fov=60.0
            )
            gt_masks, tooth_ids, valid_masks = self.generate_gt_masks_f(
                verts, faces, sem_labels, cameras, valid_teeth)
            all_gt_masks[i] = gt_masks
            all_tooth_ids[i] = tooth_ids
            all_valid_masks[i] = valid_masks
        foreground_mask = np.any(all_gt_masks, axis=1)
        background_masks = ~foreground_mask
        background_masks = np.expand_dims(background_masks, axis=1)
        gt_masks_with_bg = np.concatenate([background_masks, all_gt_masks], axis=1).astype(np.bool_)

        output_path = os.path.join(save_dir, f"{fname}.npz")
        np.savez(
            output_path,
            images=all_images,
            gt_masks=gt_masks_with_bg,
            tooth_ids=all_tooth_ids,   
            valid_masks=all_valid_masks
        )
        print(f"Saved preprocessed data to {output_path}")

def batch_process_teeth(data_dir, save_dir):
    """Batch process multiple tooth models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ToothPreprocessor(device=device)
    
    for jaw in ['upper', 'lower']:
        jaw_data_dir = os.path.join(data_dir, jaw)
        pc_dirs = sorted(glob.glob(f"{jaw_data_dir}/*"))
        for i in tqdm(range(len(pc_dirs))):
            pc_dir = pc_dirs[i]
            case = os.path.basename(pc_dir)
            
            obj_path = os.path.join(pc_dir, f"{case}_{jaw}.obj")
            label_path = os.path.join(pc_dir, f"{case}_{jaw}.json")
            
            if os.path.exists(obj_path) and os.path.exists(label_path):
                print(f"Processing {case}_{jaw}")
                processor.process_single_tooth(obj_path, label_path, save_dir, jaw_type=jaw)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument('--save_dir', required=True, type=str)
    args = parser.parse_args()

    batch_process_teeth(args.data_dir, args.save_dir)
