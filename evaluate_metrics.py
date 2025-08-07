import argparse
import os
import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
import sys

UPPER_TEETH_FDI = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
LOWER_TEETH_FDI = [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
FDI_TO_LABEL = {}
for i, (upper, lower) in enumerate(zip(UPPER_TEETH_FDI, LOWER_TEETH_FDI)):
    FDI_TO_LABEL[upper] = i + 1 
    FDI_TO_LABEL[lower] = i + 1 
GROUP_PAIRS = [(8, 9), (7, 10), (6, 11), (5, 12), (4, 13), (3, 14), (2, 15), (1, 16)]

class TeeLogger:
    def __init__(self, filepath):
        log_dir = os.path.dirname(filepath)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        self.file = open(filepath, 'w', buffering=1) 
        self.stdout = sys.stdout
    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

def simple_load_obj(path):
    verts = []
    faces = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(v) for v in line.strip().split()[1:]])
            elif line.startswith('f '):
                face = [int(v.split('/')[0]) - 1 for v in line.strip().split()[1:]]
                faces.append(face)
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)

def build_vertex_adjacency(faces, num_vertices):
    adj = [set() for _ in range(num_vertices)]
    for face in faces:
        if len(face) >= 3:
            v1, v2, v3 = face[0], face[1], face[2]
            adj[v1].update([v2, v3])
            adj[v2].update([v1, v3])
            adj[v3].update([v1, v2])
    return [list(s) for s in adj]

def get_boundary_vertices_knn(verts, vertex_labels, k=10):

    if NearestNeighbors is None:
        return None
    num_vertices = verts.shape[0]
    if num_vertices == 0:
        return np.array([], dtype=np.int32)
        
    n_neighbors = min(k + 1, num_vertices)
    nn_finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    nn_finder.fit(verts)
    _, indices = nn_finder.kneighbors(verts)

    neighbor_labels = vertex_labels[indices]
    is_boundary = np.any(neighbor_labels != neighbor_labels[:, [0]], axis=1)
    boundary_verts_indices = np.where(is_boundary)[0]

    return boundary_verts_indices.astype(np.int32)

def load_gt_labels(label_path):
    with open(label_path, 'r') as f:
        gt_data = json.load(f)
    original_labels = np.array(gt_data['labels'])
    sem_labels = np.zeros_like(original_labels)
    for i, fdi_number in enumerate(original_labels):
        if fdi_number > 0 and fdi_number in FDI_TO_LABEL:
            sem_labels[i] = FDI_TO_LABEL[fdi_number]
        else:
            sem_labels[i] = 0 
    return sem_labels

def load_pred_labels(label_path):
    return np.loadtxt(label_path, dtype=np.int64)

def calculate_metrics(pred_labels, gt_labels, verts=None, k=10):
    pred_np = np.asarray(pred_labels)
    gt_np = np.asarray(gt_labels)

    # 1. Overall Accuracy (OA)
    oa = np.sum(pred_np == gt_np) / len(gt_np)

    # 2. Boundary IoU (bIoU)
    biou = 0.0
    if verts is not None and NearestNeighbors is not None:
        gt_boundary = get_boundary_vertices_knn(verts, gt_np, k=k)
        pred_boundary = get_boundary_vertices_knn(verts, pred_np, k=k)

        if gt_boundary is not None and pred_boundary is not None:
            intersection = np.intersect1d(gt_boundary, pred_boundary).size
            union = np.union1d(gt_boundary, pred_boundary).size
            if union > 0:
                biou = intersection / union

    gt_present_labels = np.unique(gt_np)
    gt_present_labels = gt_present_labels[gt_present_labels > 0]

    if len(gt_present_labels) == 0:
        return {'mean_iou': 0.0, 'mean_dice': 0.0, 'teeth_count': 0, 'tooth_metrics': {}, 'oa': float(oa), 'biou': float(biou)}

    iou_sum = 0.0
    dice_sum = 0.0
    tooth_metrics = {}

    for tooth_id in gt_present_labels:
        gt_mask = (gt_np == tooth_id)
        pred_mask = (pred_np == tooth_id)
        
        intersection = np.sum(gt_mask & pred_mask)
        union = np.sum(gt_mask | pred_mask)
    
        iou = intersection / union if union > 0 else 0.0
        
        dice_denominator = np.sum(gt_mask) + np.sum(pred_mask)
        dice = (2.0 * intersection) / dice_denominator if dice_denominator > 0 else 0.0
        
        iou_sum += iou
        dice_sum += dice
        
        tooth_metrics[int(tooth_id)] = {
            'iou': float(iou),
            'dice': float(dice)
        }

    teeth_count = len(gt_present_labels)
    mean_iou = iou_sum / teeth_count
    mean_dice = dice_sum / teeth_count
    
    return {
        'mean_iou': float(mean_iou),
        'mean_dice': float(mean_dice),
        'teeth_count': int(teeth_count),
        'oa': float(oa),
        'biou': float(biou),
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D tooth segmentation metrics from saved label files.")
    parser.add_argument('--pred_dir', type=str, required=True, help="Root directory of the prediction results (e.g., 'inference_results').")
    parser.add_argument('--gt_dir', type=str, required=True, help="Root directory of the ground truth data.")
    parser.add_argument('--test_split_file', type=str, required=True, help="Path to a text file listing case names to evaluate.")
    parser.add_argument('--knn', type=int, default=10, help="Number of nearest neighbors (k) for bIoU calculation.")
    parser.add_argument('--log_file', type=str, default='results/metrics.txt', help="File path to save evaluation log (stdout will be tee'd).")
    
    args = parser.parse_args()

    if args.log_file:
        sys.stdout = TeeLogger(args.log_file)
    with open(args.test_split_file, 'r') as f:
        cases_to_eval = set(line.strip() for line in f if line.strip())
    all_metrics = []

    for case_id in sorted(list(cases_to_eval)):
        case_name, jaw_type = case_id.rsplit('_', 1)

        pred_label_path = os.path.join(args.pred_dir, jaw_type, f"{case_name}_{jaw_type}.txt")
        gt_label_path = os.path.join(args.gt_dir, f"{jaw_type}", case_name, f"{case_name}_{jaw_type}.json")
        obj_path = os.path.join(args.gt_dir, f"{jaw_type}", case_name, f"{case_name}_{jaw_type}.obj")

        pred_labels = load_pred_labels(pred_label_path)
        gt_labels = load_gt_labels(gt_label_path)
        
        verts, _ = simple_load_obj(obj_path)
        metrics = calculate_metrics(pred_labels, gt_labels, verts=verts, k=args.knn)
        all_metrics.append(metrics)
        
        print(f"  Processed '{case_id}': Mean IoU = {metrics['mean_iou']:.4f}, Mean Dice = {metrics['mean_dice']:.4f}, OA = {metrics['oa']:.4f}, bIoU = {metrics['biou']:.4f}")

    print("\n" + "="*50)
    print("           METRICS SUMMARY")
    print("="*50)

    def print_summary(name, metrics_list):
        avg_iou = np.mean([m['mean_iou'] for m in metrics_list])
        avg_dice = np.mean([m['mean_dice'] for m in metrics_list])
        avg_oa = np.mean([m['oa'] for m in metrics_list])
        avg_biou = np.mean([m['biou'] for m in metrics_list])

        print(f"\n{name} ({len(metrics_list)} cases):")
        print(f"  - Average Mean IoU        (mIoU): {avg_iou:.4f}")
        print(f"  - Average Mean Dice             : {avg_dice:.4f}")
        print(f"  - Average Overall Accuracy (OA) : {avg_oa:.4f}")
        print(f"  - Average Boundary IoU   (bIoU) : {avg_biou:.4f}")

    print_summary("Overall", all_metrics)
    print("="*50)

if __name__ == '__main__':
    main() 