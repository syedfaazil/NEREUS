"""
Dataset Preparation Script
Prepares the zooplankton dataset for training YOLOv5 and MobileNetV3 models.

Optional flags for HPC workflow:
  --resize 224          Resize images to NxN during preparation (reduces archive size).
  --output-archive      After preparation, creates prepared_dataset.tar.gz for HF Hub upload.
"""

import sys
import tarfile
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import json
import shutil
import concurrent.futures

# ── Load config (optional) ────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
try:
    import config as _cfg
    _RESIZE_TO       = getattr(_cfg, "RESIZE_TO", None)
    _SAMPLE_FRACTION = getattr(_cfg, "SAMPLE_FRACTION", 1.0)
    _HF_DATASET_REPO = getattr(_cfg, "HF_DATASET_REPO", "")
except ImportError:
    _RESIZE_TO       = None
    _SAMPLE_FRACTION = 1.0
    _HF_DATASET_REPO = ""

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class DatasetPreparer:
    """Prepares zooplankton dataset for model training"""
    
    def __init__(self, root_dir: str, output_dir: str = None, resize_to: int = None):
        """
        Initialize dataset preparer
        
        Args:
            root_dir:  Root directory of the dataset codebase (where zip is saved)
            output_dir: Output directory for extracted images and prepared variants (e.g. HPC scratch drive)
            resize_to: If set, resize images to this square size (e.g. 224) during prep.
        """
        self.root_dir = Path(root_dir)
        self.output_root = Path(output_dir) if output_dir else self.root_dir
        self.images_dir = self.output_root / 'individual_images'
        self.taxonomy_file = self.root_dir / 'taxonomy_descriptor_zooscan.csv'
        self.resize_to = resize_to
        
        # Output directories
        self.output_dir = self.output_root / 'prepared_dataset'
        self.yolo_dir = self.output_dir / 'yolo'
        self.classification_dir = self.output_dir / 'classification'
        
        # Class mapping
        self.class_to_id = {}
        self.id_to_class = {}
    
    def ensure_raw_dataset(self, repo_id: str):
        """Download and extract raw dataset if it doesn't exist."""
        if self.images_dir.exists() and any(self.images_dir.iterdir()):
            return
            
        print(f"\nRaw dataset not found at {self.images_dir}.")
        
        local_zip = self.output_root / 'individual_images.zip'
        zip_path = str(local_zip)
        
        if not local_zip.exists():
            if not repo_id:
                raise ValueError("images_dir not found, local zip not found, and no HF_DATASET_REPO provided to download from.")
                
            print(f"Downloading raw dataset from Hugging Face Hub: {repo_id} to {self.output_root}...")
            try:
                from huggingface_hub import hf_hub_download
                zip_path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename="individual_images.zip",
                    local_dir=str(self.output_root)
                )
            except ImportError:
                raise ImportError("huggingface_hub is required to download the dataset. Please run `pip install huggingface_hub`.")
        else:
            print(f"Found local zip file: {local_zip}")
            
        print(f"Extracting dataset to {self.output_root}...")
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.output_root)
            print("Extraction complete.")
        except Exception as e:
            print(f"Failed to extract dataset: {e}")
            raise

    def load_taxonomy(self):
        """Load taxonomy information"""
        print("Loading taxonomy information...")
        
        # Get all class directories
        class_dirs = [d for d in self.images_dir.iterdir() if d.is_dir()]
        
        # Create class mapping
        for i, class_dir in enumerate(sorted(class_dirs)):
            class_name = class_dir.name
            self.class_to_id[class_name] = i
            self.id_to_class[i] = class_name
        
        print(f"Found {len(self.class_to_id)} classes")
        
        # Save class mapping
        mapping_file = self.output_dir / 'class_mapping.json'
        mapping_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(mapping_file, 'w') as f:
            json.dump({
                'class_to_id': self.class_to_id,
                'id_to_class': self.id_to_class,
                'num_classes': len(self.class_to_id)
            }, f, indent=2)
        
        print(f"Class mapping saved to {mapping_file}")
        
        # Build hierarchy
        self.hierarchy = {}
        for class_name in self.class_to_id.keys():
            # Many plankton datasets use double underscore or dash for separation
            if '__' in class_name:
                higher_taxa = class_name.split('__')[0]
            elif '_' in class_name:
                higher_taxa = class_name.split('_')[0]
            else:
                higher_taxa = class_name
                
            # specific fix for 'badfocus' or 'detritus'
            if 'badfocus' in class_name.lower() or 'artefact' in class_name.lower() or 'bubble' in class_name.lower():
                higher_taxa = 'artifact'
            elif 'detritus' in class_name.lower():
                higher_taxa = 'artifact'
                
            self.hierarchy[class_name] = higher_taxa
            
        # Get unique higher taxa
        self.higher_taxa_list = sorted(list(set(self.hierarchy.values())))
        self.higher_to_id = {name: i for i, name in enumerate(self.higher_taxa_list)}
        
        hierarchy_file = self.output_dir / 'hierarchy.json'
        with open(hierarchy_file, 'w') as f:
            json.dump({
                'class_to_higher': self.hierarchy,
                'higher_to_id': self.higher_to_id,
                'id_to_higher': {i: name for name, i in self.higher_to_id.items()},
                'num_higher_classes': len(self.higher_to_id)
            }, f, indent=2)
        print(f"Hierarchy mapping saved to {hierarchy_file} ({len(self.higher_to_id)} higher-level taxa)")
    
    def prepare_classification_dataset(self, 
                                      train_split: float = 0.7,
                                      val_split: float = 0.15,
                                      test_split: float = 0.15,
                                      sample_fraction: float = 1.0):
        """
        Prepare dataset for classification training
        
        Args:
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
            sample_fraction: Fraction of dataset to use (0.0 to 1.0)
        """
        print("\nPreparing classification dataset...")
        
        # Create directories
        for split in ['train', 'val', 'test']:
            split_dir = self.classification_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Collect images per class
        class_images = {}
        for class_name, class_id in self.class_to_id.items():
            class_dir = self.images_dir / class_name
            if not class_dir.exists(): continue
            imgs = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            if len(imgs) > 0:
                class_images[class_name] = imgs

        # 2. Group rare classes
        rare_images = []
        regular_classes = {}
        for cname, imgs in class_images.items():
            if len(imgs) < 5:
                rare_images.extend(imgs)
            else:
                regular_classes[cname] = imgs
                
        if rare_images:
            regular_classes['rare_species'] = rare_images
            if 'rare_species' not in self.class_to_id:
                new_id = max(self.id_to_class.keys()) + 1 if self.id_to_class else 0
                self.class_to_id['rare_species'] = new_id
                self.id_to_class[new_id] = 'rare_species'
                # Re-save mapping to include rare_species
                mapping_file = self.output_dir / 'class_mapping.json'
                with open(mapping_file, 'w') as f:
                    json.dump({
                        'class_to_id': self.class_to_id,
                        'id_to_class': self.id_to_class,
                        'num_classes': len(self.class_to_id)
                    }, f, indent=2)

        # 3. Process classes
        stats = {'train': 0, 'val': 0, 'test': 0}
        for class_name, images in tqdm(regular_classes.items(), desc="Processing classes"):
            np.random.shuffle(images)
            
            # Oversample rare_species so the model actually learns its features
            if class_name == 'rare_species' and len(images) < 50:
                repeats = (50 // len(images)) + 1
                images = (images * repeats)[:50]
            
            # Subsample if requested
            if sample_fraction < 1.0:
                n_sample = max(1, int(len(images) * sample_fraction))
                images = images[:n_sample]
            
            # Split
            n_train = max(1, int(len(images) * train_split))
            n_val = max(1, int(len(images) * val_split))
            if len(images) < 3: # extreme fallback
                n_train, n_val = len(images), 0
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train + n_val]
            test_images = images[n_train + n_val:]
            
            # Copy (and optionally resize) images to respective directories
            
            def process_image(args):
                split, i, img_path, class_name = args
                split_class_dir = self.classification_dir / split / class_name
                # Use index prefix to handle duplicate filenames from oversampling
                dst = split_class_dir / f"{i:04d}_{img_path.name}"
                if self.resize_to is not None:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        img = cv2.resize(img, (self.resize_to, self.resize_to),
                                         interpolation=cv2.INTER_AREA)
                        cv2.imwrite(str(dst), img,
                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
                    else:
                        shutil.copy2(img_path, dst)
                else:
                    shutil.copy2(img_path, dst)
                return split

            tasks = []
            for split, split_images in [('train', train_images), 
                                       ('val', val_images), 
                                       ('test', test_images)]:
                if len(split_images) == 0:
                    continue  # Don't create empty class dirs
                split_class_dir = self.classification_dir / split / class_name
                split_class_dir.mkdir(parents=True, exist_ok=True)
                
                for i, img_path in enumerate(split_images):
                    tasks.append((split, i, img_path, class_name))
            
            if tasks:
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    for split in executor.map(process_image, tasks):
                        stats[split] += 1
        
        print(f"\nClassification dataset prepared:")
        print(f"  Train: {stats['train']} images")
        print(f"  Val: {stats['val']} images")
        print(f"  Test: {stats['test']} images")
        print(f"  Location: {self.classification_dir}")
    
    def prepare_detection_dataset(self,
                                  train_split: float = 0.7,
                                  val_split: float = 0.15,
                                  test_split: float = 0.15,
                                  create_composite: bool = True,
                                  images_per_composite: int = 5,
                                  sample_fraction: float = 1.0):
        """
        Prepare dataset for YOLO detection training
        Creates composite images with multiple specimens
        
        Args:
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
            create_composite: Whether to create composite images
            images_per_composite: Number of specimens per composite
            sample_fraction: Fraction of dataset to use
        """
        print("\nPreparing detection dataset...")
        
        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (self.yolo_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.yolo_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        if create_composite:
            self._create_composite_images(train_split, val_split, test_split, 
                                         images_per_composite, sample_fraction)
        else:
            self._create_single_object_dataset(train_split, val_split, test_split, sample_fraction)
        
        # Create data.yaml for YOLO
        self._create_yolo_yaml()
    
    def _create_composite_images(self, train_split, val_split, test_split, 
                                images_per_composite, sample_fraction=1.0):
        """Create composite images with multiple specimens"""
        print("Creating composite images...")
        
        # Collect all images
        all_images = []
        for class_name, class_id in self.class_to_id.items():
            class_dir = self.images_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                if sample_fraction < 1.0:
                    np.random.shuffle(images)
                    images = images[:int(len(images) * sample_fraction)]
                
                for img_path in images:
                    all_images.append((img_path, class_id))
        
        np.random.shuffle(all_images)
        
        # Split
        n_train = int(len(all_images) * train_split)
        n_val = int(len(all_images) * val_split)
        
        splits = {
            'train': all_images[:n_train],
            'val': all_images[n_train:n_train + n_val],
            'test': all_images[n_train + n_val:]
        }
        
        # Create composite images for each split
        for split_name, split_images in splits.items():
            print(f"Creating {split_name} composites...")
            
            # Create composites
            n_composites = len(split_images) // images_per_composite
            
            for i in tqdm(range(n_composites), desc=f"{split_name} composites"):
                # Select random images for this composite
                composite_images = split_images[i * images_per_composite:
                                               (i + 1) * images_per_composite]
                
                # Create composite
                composite_img, annotations = self._create_single_composite(
                    composite_images, canvas_size=640
                )
                
                # Save image
                img_name = f"composite_{i:06d}.jpg"
                img_path = self.yolo_dir / split_name / 'images' / img_name
                cv2.imwrite(str(img_path), composite_img)
                
                # Save annotations in YOLO format
                label_path = self.yolo_dir / split_name / 'labels' / f"composite_{i:06d}.txt"
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann['class']} {ann['x_center']} {ann['y_center']} "
                               f"{ann['width']} {ann['height']}\n")
        
        print("Composite images created successfully!")
    
    def _create_single_composite(self, images_info, canvas_size=640):
        """Create a single composite image from a list of (image_path, class_id) tuples."""
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        annotations = []
        
        n_imgs = len(images_info)
        if n_imgs == 0:
            return canvas, annotations
            
        grid_cols = int(np.ceil(np.sqrt(n_imgs)))
        grid_rows = int(np.ceil(n_imgs / grid_cols))
        
        cell_w = canvas_size // grid_cols
        cell_h = canvas_size // grid_rows
        
        for idx, (img_path, class_id) in enumerate(images_info):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            row = idx // grid_cols
            col = idx % grid_cols
            
            x1 = col * cell_w
            y1 = row * cell_h
            
            h, w = img.shape[:2]
            scale = min((cell_w - 4) / max(w, 1), (cell_h - 4) / max(h, 1))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            start_x = x1 + (cell_w - new_w) // 2
            start_y = y1 + (cell_h - new_h) // 2
            
            canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
            
            x_center = (start_x + new_w / 2) / canvas_size
            y_center = (start_y + new_h / 2) / canvas_size
            width = new_w / canvas_size
            height = new_h / canvas_size
            
            annotations.append({
                'class': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
            
        return canvas, annotations
    
    def _create_single_object_dataset(self, train_split, val_split, test_split, sample_fraction=1.0):
        """Create dataset with single objects (simpler approach)"""
        print("Creating single-object detection dataset...")
        
        # Similar to classification but with YOLO annotations
        for class_name, class_id in tqdm(self.class_to_id.items(), desc="Processing"):
            class_dir = self.images_dir / class_name
            
            if not class_dir.exists():
                continue
            
            images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            if len(images) == 0:
                continue
            
            np.random.shuffle(images)
            
            # Subsample
            if sample_fraction < 1.0:
                images = images[:int(len(images) * sample_fraction)]
            
            # Split
            n_train = int(len(images) * train_split)
            n_val = int(len(images) * val_split)
            
            splits = {
                'train': images[:n_train],
                'val': images[n_train:n_train + n_val],
                'test': images[n_train + n_val:]
            }
            
            for split_name, split_images in splits.items():
                for img_path in split_images:
                    # Copy image
                    dst_img = self.yolo_dir / split_name / 'images' / img_path.name
                    shutil.copy2(img_path, dst_img)
                    
                    # Create annotation (object fills entire image)
                    label_path = self.yolo_dir / split_name / 'labels' / f"{img_path.stem}.txt"
                    with open(label_path, 'w') as f:
                        # Center of image, full size
                        f.write(f"{class_id} 0.5 0.5 0.9 0.9\n")
                        
    def _create_yolo_yaml(self):
        """Create data.yaml for YOLO training"""
        import yaml
        
        yaml_path = self.yolo_dir / 'data.yaml'
        
        # Sort classes by ID to ensure correct mapping
        sorted_classes = [self.id_to_class[i] for i in range(len(self.id_to_class))]
        
        data = {
            'path': str(self.yolo_dir),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.class_to_id),
            'names': sorted_classes
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
            
        print(f"Created {yaml_path}")
    
    def create_archive(self) -> Path:
        """
        Compress the prepared_dataset directory into a single .tar.gz archive
        suitable for uploading to Hugging Face Hub and pulling on HPC compute nodes.

        Returns:
            Path to the created archive.
        """
        archive_path = self.root_dir / 'prepared_dataset.tar.gz'
        print(f"\nCreating archive: {archive_path}")
        print("This may take a few minutes...")

        with tarfile.open(archive_path, 'w:gz') as tar:
            tar.add(self.output_dir, arcname='prepared_dataset')

        size_gb = archive_path.stat().st_size / 1e9
        print(f"Archive created: {archive_path} ({size_gb:.2f} GB)")
        print("Upload it with:")
        print(f"  python scripts/upload_prepared.py --token $HF_TOKEN --repo yourname/planktonai-dataset")
        return archive_path


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare Zooplankton dataset')
    parser.add_argument('--sample', type=float, default=_SAMPLE_FRACTION,
                        help=f'Fraction of dataset to use (default: {_SAMPLE_FRACTION} from config.py)')
    parser.add_argument('--resize', type=int, default=_RESIZE_TO,
                        help=f'Resize images to NxN (default: {_RESIZE_TO} from config.py)')
    parser.add_argument('--output-archive', action='store_true',
                        help='Create prepared_dataset.tar.gz after preparation')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to extract and prepare datasets (use for fast scratch disk on HPC)')
    args = parser.parse_args()
    
    print("=" * 50)
    print("Zooplankton Dataset Preparation")
    print("=" * 50)
    
    if args.sample < 1.0:
        print(f"NOTE: Using {args.sample*100:.0f}% of the dataset")
    if args.resize:
        print(f"NOTE: Images will be resized to {args.resize}x{args.resize}px")
    if args.output_archive:
        print("NOTE: Will create prepared_dataset.tar.gz after preparation")
    
    # Initialize preparer
    root_dir = Path(__file__).parent.parent
    preparer = DatasetPreparer(root_dir, output_dir=args.output_dir, resize_to=args.resize)
    
    # Early exit if prepared dataset is already present
    if (preparer.classification_dir / 'train').exists() and preparer.yolo_dir.exists():
        print(f"\nPrepared dataset already exists at {preparer.output_dir}.")
        print("Skipping download and preparation completely.")
        return
    
    # Ensure raw dataset is present
    preparer.ensure_raw_dataset(_HF_DATASET_REPO)
    
    # Load taxonomy
    preparer.load_taxonomy()
    
    # Get statistics (method removed — add back if needed)
    # preparer.get_dataset_statistics()
    
    # Prepare datasets
    print("\n" + "=" * 50)
    print("Preparing datasets...")
    print("=" * 50)
    
    # Classification dataset
    preparer.prepare_classification_dataset(
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        sample_fraction=args.sample
    )
    
    # Detection dataset (with composite images)
    preparer.prepare_detection_dataset(
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        create_composite=True,
        images_per_composite=5,
        sample_fraction=args.sample
    )
    
    print("\n" + "=" * 50)
    print("Dataset preparation completed!")
    print("=" * 50)
    print(f"\nOutput directory: {preparer.output_dir}")
    print(f"  - Classification: {preparer.classification_dir}")
    print(f"  - Detection (YOLO): {preparer.yolo_dir}")
    
    # Optionally create archive for HPC upload
    if args.output_archive:
        print("\n" + "=" * 50)
        preparer.create_archive()


if __name__ == '__main__':
    main()
