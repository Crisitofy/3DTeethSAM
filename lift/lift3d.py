import numpy as np
import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    MeshRasterizer,
    RasterizationSettings
)
from pytorch3d.structures import Meshes
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

class Projector:
    def __init__(self, device='cuda', image_size=512, camera_distance=70):
        self.device = device
        self.image_size = image_size
        self.camera_distance = camera_distance
        self.views = [
            [10, 0],  
            [10, 45],  
            [10, 315], 
            [320, 0],  
            [315, 45], 
            [315, 315], 
            [60, 0],    
        ]
        self.cameras = []
        for view in self.views:
            R, T = look_at_view_transform(
                dist=self.camera_distance,
                elev=view[0],
                azim=view[1],
                device=self.device
            )
            self.cameras.append(FoVPerspectiveCameras(
                device=self.device,
                R=R,
                T=T,
                znear=0.01,
                zfar=100.0,
                aspect_ratio=1.0,
                fov=60.0
            ))
    
    def project(self, vertices, faces=None):
        vertices = torch.tensor(vertices, device=self.device)
        faces = torch.tensor(faces, device=self.device) if faces is not None else None
        num_vertices = len(vertices)
        num_views = len(self.views)

        projected_points = torch.zeros((num_views, num_vertices, 2), device=self.device)
        visibility = torch.zeros((num_views, num_vertices), dtype=torch.bool, device=self.device)
        depths = torch.zeros((num_views, num_vertices), device=self.device)
        
        if faces is not None:
            mesh = Meshes(verts=[vertices], faces=[faces])
            face_vertices = faces.flatten()
            face_indices = torch.arange(len(faces), device=self.device).repeat_interleave(3)
            indices = torch.stack([face_vertices, face_indices])
            values = torch.ones(len(face_vertices), device=self.device)
            vert_to_faces_map = torch.sparse_coo_tensor(
                indices, values, 
                (num_vertices, len(faces))
            ).coalesce()
        
        for view_idx, camera in enumerate(self.cameras):
            projected = camera.transform_points_screen(
                vertices.unsqueeze(0),  # [1, N, 3]
                image_size=(self.image_size, self.image_size)
            )
            projected_points[view_idx] = projected[0, :, :2]
            depths[view_idx] = projected[0, :, 2]

            if faces is None:
                visibility[view_idx] = torch.ones(num_vertices, dtype=torch.bool, device=self.device)
                continue
            raster_settings = RasterizationSettings(
                image_size=self.image_size,
                blur_radius=0.0,
                faces_per_pixel=1,
                cull_backfaces=True,
            )
            rasterizer = MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings
            )
            fragments = rasterizer(mesh)
            face_idxs = fragments.pix_to_face[0, :, :, 0]  # [H, W]
            visible_face_idxs = face_idxs[face_idxs >= 0].unique()

            if len(visible_face_idxs) > 0:
                visible_faces_mask = torch.zeros(len(faces), dtype=torch.float32, device=self.device)
                visible_faces_mask[visible_face_idxs] = 1.0
                vert_visibility = torch.sparse.mm(
                    vert_to_faces_map, 
                    visible_faces_mask.unsqueeze(1)
                ).squeeze(1)
                visibility[view_idx] = (vert_visibility > 0)
        return projected_points, visibility, depths, None

    def assign_masks(self, projected_points, visibility, masks, valid_masks):

        num_views, num_points, _ = projected_points.shape
        num_masks = masks.shape[1]
        assignments = torch.zeros((num_views, num_points, num_masks), dtype=torch.float32, device=self.device)
        
        for view_idx in range(num_views):
            view_points = projected_points[view_idx]  # [N, 2]
            view_masks = torch.tensor(masks[view_idx], device=self.device)  # [M, H, W]
            view_valid = valid_masks[view_idx]  # [M]
            view_visibility = visibility[view_idx]  # [N]
            
            valid_points = (view_points[:, 0] >= 0) & (view_points[:, 0] < self.image_size - 1) & \
                          (view_points[:, 1] >= 0) & (view_points[:, 1] < self.image_size - 1) & \
                          view_visibility
            valid_points_idx = valid_points.nonzero().squeeze(1)
            
            if len(valid_points_idx) > 0:
                valid_coords = view_points[valid_points_idx]  # [K, 2]
                scaled_coords = valid_coords.clone()
                scaled_coords[:, 0] = 2.0 * (scaled_coords[:, 0] / (self.image_size - 1)) - 1.0
                scaled_coords[:, 1] = 2.0 * (scaled_coords[:, 1] / (self.image_size - 1)) - 1.0
                grid = scaled_coords.view(1, -1, 1, 2)  # [1, K, 1, 2]

                for mask_idx in range(num_masks):
                    if not view_valid[mask_idx]:
                        continue
                    mask = view_masks[mask_idx].unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]
                    sampled_mask = F.grid_sample(
                        mask, 
                        grid, 
                        mode='bilinear', 
                        padding_mode='border', 
                        align_corners=True
                    )  # [1, 1, K, 1]
                    mask_values = sampled_mask.view(-1)
                    assignments[view_idx, valid_points_idx, mask_idx] = (mask_values > 0.5).float()

        return assignments
    
class Labeler:
    def __init__(self, device='cuda'):
        self.device = device
        
    def vote(self, assignments, tooth_ids, visibility=None, depths=None, view_weights=None, tooth_bias=2.0):
        num_views, num_points, num_masks = assignments.shape
        device = self.device
        
        tooth_votes = torch.zeros((num_points, 16), device=device)
        background_votes = torch.zeros(num_points, device=device)

        base_weights = torch.ones((num_views, num_points), device=device)
        if view_weights is None:
            view_weights_tensor = torch.ones(num_views, device=device)
        else:
            view_weights_tensor = torch.tensor(view_weights, dtype=torch.float32, device=device)
        base_weights *= view_weights_tensor.view(-1, 1)

        if depths is not None:
            depth_weights = 1.0 / depths.clamp(min=1e-6)
            for v_idx in range(num_views):
                v_depths = depth_weights[v_idx]
                min_d, max_d = v_depths.min(), v_depths.max()
                if max_d > min_d:
                    norm_depth_weights = 1.0 + (v_depths - min_d) / (max_d - min_d)
                    base_weights[v_idx] *= norm_depth_weights
        
        if visibility is not None:
            base_weights *= visibility.float()

        for view_idx in range(num_views):
            view_assignments = assignments[view_idx]  # [N, M]
            view_tooth_ids = tooth_ids[view_idx]      # [M]
            view_weights = base_weights[view_idx]    # [N]

            prob_sum_per_point = view_assignments.sum(dim=1)  # [N]
            background_prob = (1.0 - prob_sum_per_point).clamp(min=0.0)
            background_votes += view_weights * background_prob
            vote_divisor = prob_sum_per_point.clamp(min=1e-6)

            for mask_idx in range(num_masks):
                tooth_id = view_tooth_ids[mask_idx].item()
                if tooth_id > 0:
                    prob_this_mask = view_assignments[:, mask_idx] # [N]
                    vote_for_this_tooth = view_weights * (prob_this_mask / vote_divisor)
                    tooth_votes[:, tooth_id - 1] += vote_for_this_tooth
        
        max_tooth_votes, best_tooth_indices = torch.max(tooth_votes, dim=1)
        is_tooth_mask = (max_tooth_votes * tooth_bias) > background_votes
        final_labels = torch.zeros(num_points, dtype=torch.long, device=device)
        final_labels[is_tooth_mask] = best_tooth_indices[is_tooth_mask] + 1
        
        return final_labels.cpu().numpy()
    
    def build_adjacency_matrix(self, vertices, faces, device):
        num_vertices = len(vertices)
        faces_tensor = torch.tensor(faces, device=device) if not isinstance(faces, torch.Tensor) else faces
        i = torch.cat([
            faces_tensor[:, 0], faces_tensor[:, 0], 
            faces_tensor[:, 1], faces_tensor[:, 1], 
            faces_tensor[:, 2], faces_tensor[:, 2]
        ])
        j = torch.cat([
            faces_tensor[:, 1], faces_tensor[:, 2], 
            faces_tensor[:, 0], faces_tensor[:, 2], 
            faces_tensor[:, 0], faces_tensor[:, 1]
        ])
        indices = torch.stack([i, j])
        values = torch.ones(indices.shape[1], device=device)
        adj_matrix_sparse = torch.sparse_coo_tensor(indices, values, (num_vertices, num_vertices)).coalesce()
        adj_matrix_sparse = (adj_matrix_sparse + adj_matrix_sparse.transpose(0,1)).coalesce()
        adj_matrix_sparse = torch.sparse_coo_tensor(
            adj_matrix_sparse.indices(), 
            torch.ones_like(adj_matrix_sparse.values()), 
            adj_matrix_sparse.shape
        ).coalesce()

        return adj_matrix_sparse

    def retain_largest_component_per_tooth(self, vertices, faces, labels, logger=None):
        cleaned_labels = labels.copy()
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels > 0]
        
        if len(unique_labels) == 0:
            return cleaned_labels
        adj_matrix_torch = self.build_adjacency_matrix(vertices, faces, self.device)
        indices_cpu = adj_matrix_torch._indices().cpu().numpy()
        values_cpu = adj_matrix_torch._values().cpu().numpy()
        adj_scipy = csr_matrix(
            (values_cpu, (indices_cpu[0], indices_cpu[1])), 
            shape=(len(vertices), len(vertices))
        )
        for tooth_label in unique_labels:
            tooth_mask = (labels == tooth_label)
            tooth_indices = np.where(tooth_mask)[0]
            if len(tooth_indices) < 10:
                continue
            tooth_adj_subgraph = adj_scipy[tooth_indices, :][:, tooth_indices]
            n_components, component_labels = connected_components(
                tooth_adj_subgraph, directed=False
            )
            if n_components > 1:
                component_sizes = Counter(component_labels)
                main_component_id = max(component_sizes, key=component_sizes.get)
                for comp_id, comp_size in component_sizes.items():
                    if comp_id != main_component_id:
                        comp_local_indices = np.where(component_labels == comp_id)[0]
                        comp_global_indices = tooth_indices[comp_local_indices]
                        cleaned_labels[comp_global_indices] = 0
        return cleaned_labels

    def apply_graph_cut_smoothing(self, vertices, faces, initial_labels, logger=None):  
        smoothed_labels = initial_labels.copy()
        try:
            from lift.graph_cuts.smooth import Teeth
            tooth_labels_to_process = list(range(1, 17))
            teeth = Teeth(vertices, faces)
            teeth.v_labels = smoothed_labels
            teeth.update_t_labels()

            for lbl in tooth_labels_to_process:
                if lbl not in teeth.labels:
                    continue
                teeth.fuzzy_cluster(lbl)

            teeth.update_v_labels()
            smoothed_labels = teeth.v_labels

        except Exception as e:
            if logger:
                logger.error(f"graph_cuts error: {e}", exc_info=True)
            return initial_labels
        
        return smoothed_labels

    def complete_segmentation_pipeline(self, vertices, faces, initial_labels, logger=None):
        current_labels = initial_labels.copy()
        current_labels = self.retain_largest_component_per_tooth(
            vertices, faces, current_labels, logger
        )
        final_labels = self.apply_graph_cut_smoothing(
            vertices, faces, current_labels, logger
        )
        return final_labels
    