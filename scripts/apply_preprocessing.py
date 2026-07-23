"""
Standalone Preprocessing Script
Applies the COMPLETE preprocessing pipeline:
1. Standardization (Resize/Format)
2. Enhancement (CLAHE, Denoising)
3. Artifact Removal (Bubbles, Debris)
4. Watershed Segmentation (Separate overlaps)

Uses PARALLEL PROCESSING to maximize speed.
Output: 'individual_images_preprocessed'
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import argparse
from multiprocessing import Pool, cpu_count
import functools

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from preprocessing import ImageStandardizer, ImageEnhancer, ArtifactRemover, WatershedSegmenter

def process_single_image(args):
    """
    Process a single image - function wrapper for pool.map
    args: (img_path, output_dir, relative_path)
    """
    img_path, output_root, relative_folder = args
    
    try:
        # Re-initialize preprocessors properly inside the worker process
        # to avoid pickling issues with some OpenCV objects if any
        # (Though these classes seem safe, it's good practice)
        standardizer = ImageStandardizer()
        enhancer = ImageEnhancer()
        artifact_remover = ArtifactRemover()
        watershed = WatershedSegmenter()
        
        # 1. Load
        image = cv2.imread(str(img_path))
        if image is None:
            return False, f"Failed to load {img_path.name}"
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # 2. Enhance
        enhanced = enhancer.adaptive_enhancement(gray)
        
        # 3. Remove Artifacts
        cleaned = artifact_remover.remove_artifacts(enhanced)
        
        # 4. Watershed Segmentation (Mask background)
        segmented_img, masks = watershed.segment(cleaned, return_masks=True)
        
        final_img = cleaned
        if masks:
            full_mask = np.zeros_like(cleaned)
            for m in masks:
                full_mask = cv2.bitwise_or(full_mask, m)
            final_img = cv2.bitwise_and(cleaned, cleaned, mask=full_mask)
        
        # Save
        # Make sure directory exists (race condition safe)
        out_folder = output_root / relative_folder
        out_folder.mkdir(parents=True, exist_ok=True)
        
        out_path = out_folder / img_path.name
        cv2.imwrite(str(out_path), final_img)
        
        return True, None
        
    except Exception as e:
        return False, f"Error processing {img_path.name}: {str(e)}"

def process_dataset(input_dir: str, output_dir: str, num_workers: int = None):
    """
    Process all images in the dataset using parallel processing
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' not found.")
        return
        
    if num_workers is None:
        num_workers = cpu_count()

    print("=" * 50)
    print("Zooplankton Full Preprocessing Pipeline (Parallel)")
    print("=" * 50)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Workers: {num_workers}")

    # Gather all tasks
    tasks = []
    
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    print(f"Scanning {len(class_dirs)} classes...")
    
    for class_dir in class_dirs:
        # Find images
        images = []
        for ext in ['*.jpg', '*.png', '*.tif', '*.tiff', '*.jpeg']:
            images.extend(list(class_dir.glob(ext)))
            
        # Create task tuples
        for img_path in images:
            # Task: (path_to_image, output_root, class_name)
            tasks.append((img_path, output_path, class_dir.name))
            
    total_images = len(tasks)
    print(f"Found {total_images} images to process.")
    
    # Run in parallel
    processed_count = 0
    error_count = 0
    
    print("\nStarting parallel processing...")
    
    with Pool(processes=num_workers) as pool:
        # Use tqdm to show progress bar
        results = list(tqdm(pool.imap(process_single_image, tasks), 
                          total=total_images,
                          desc="Processing"))
        
        # Analyze results
        for success, message in results:
            if success:
                processed_count += 1
            else:
                error_count += 1
                # print(message) # Optional: print errors
                
    print("\n" + "=" * 50)
    print("Preprocessing Completed")
    print("=" * 50)
    print(f"Processed: {processed_count}")
    print(f"Errors: {error_count}")
    print(f"Output saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Apply full preprocessing to dataset (Parallel)')
    parser.add_argument('--input', type=str, default='individual_images',
                       help='Input directory containing class subfolders')
    parser.add_argument('--output', type=str, default='individual_images_preprocessed',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: all CPUs)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    root = Path(__file__).parent.parent
    in_dir = root / args.input
    out_dir = root / args.output
    
    # On Windows, this protection is needed for multiprocessing
    process_dataset(in_dir, out_dir, args.workers)
