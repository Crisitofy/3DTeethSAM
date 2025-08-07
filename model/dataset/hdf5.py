import h5py
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import argparse

def convert_to_hdf5(data_dir, output_file, compression="gzip", views=7):
    """
    Converts a directory of .npz files into a single HDF5 file.
    
    Args:
        data_dir: The source data directory containing .npz files.
        output_file: The path for the output HDF5 file.
        compression: The compression algorithm to use ("lzf" is fast with low compression, 
                     "gzip" is slower with high compression).
    """
    data_dir = Path(data_dir)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all .npz files
    npz_files = list(data_dir.glob("*.npz"))
    
    samples = []
    for npz_path in tqdm(npz_files, desc="Scanning .npz files"):
        case_name = npz_path.stem
        # Each .npz file contains multiple views
        for view_idx in range(views):
             samples.append({
                'npz_path': str(npz_path),
                'view_idx': view_idx,
                'case_name': case_name
            })
    
    print(f"Found {len(samples)} samples ({len(npz_files)} files x {views} views), starting conversion...")
    
    # Create HDF5 file and save data
    with h5py.File(output_file, 'w') as f:
        # Create data groups
        images_group = f.create_group('images')
        gt_masks_group = f.create_group('gt_masks')
        metadata_group = f.create_group('metadata')
        
        # Store sample index
        sample_index = []
        
        # Cache npz data to avoid redundant loading
        cached_npz = {}
        
        # Process each sample
        for idx, sample in enumerate(tqdm(samples, desc="Converting data")):
            npz_path = sample['npz_path']
            view_idx = sample['view_idx']
            case_name = sample['case_name']
            sample_id = f"{case_name}_view{view_idx}"
            
            # Record sample index information
            sample_index.append({
                'id': sample_id,
                'case_name': case_name,
                'view_idx': view_idx
            })
            
            # Load npz file (with caching)
            if npz_path not in cached_npz:
                cached_npz[npz_path] = np.load(npz_path)
            npz_data = cached_npz[npz_path]
            
            # 1. Process images
            image_array = npz_data['images'][view_idx] # Already [H, W, 3] float32
            images_group.create_dataset(
                sample_id, 
                data=image_array,
                compression=compression
            )
            
            # 2. Process GT masks including background
            gt_masks = npz_data['gt_masks'][view_idx]
            gt_masks_group.create_dataset(
                sample_id, 
                data=gt_masks,
                compression=compression
            )
            
            # 4. Process metadata
            # Save valid mask status
            metadata_group.create_dataset(
                f"{sample_id}_valid_masks", 
                data=npz_data['valid_masks'][view_idx]
            )
            
            # Save tooth IDs
            metadata_group.create_dataset(
                f"{sample_id}_tooth_ids", 
                data=npz_data['tooth_ids'][view_idx]
            )

        # Save sample index to metadata
        f.create_dataset('sample_index', data=json.dumps(sample_index))
    
    print(f"Conversion complete! Data saved to {output_file}")
    return output_file
