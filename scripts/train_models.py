"""
Model Training Script
Train YOLOv5n detector and MobileNetV3 classifier
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import argparse
import json
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from detection import YOLODetector
from classification import MobileNetClassifier, TwoStageBinaryFilter


def train_detector(data_yaml: str, 
                  epochs: int = 100,
                  batch_size: int = 16,
                  device: str = 'cuda',
                  workers: int = 8,
                  amp: bool = True):
    """
    Train YOLOv5n detector
    
    Args:
        data_yaml: Path to YOLO data.yaml
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on
        workers: Number of workers
        amp: Use AMP
    """
    print("=" * 50)
    print("Training YOLO26n Detector")
    print("=" * 50)
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'

    # Initialize detector
    detector = YOLODetector(device=device)
    
    # Print model info
    info = detector.get_model_info()
    print(f"\nModel Information:")
    print(f"  Type: {info['model_type']}")
    print(f"  Parameters: {info['parameters']:,}")
    print(f"  Device: {info['device']}")
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Data: {data_yaml}")
    
    results = detector.train(
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=416,  # Downscaled for Raspberry Pi 5 CPU real-time inference
        batch_size=batch_size,
        project='runs/detect',
        name='zooplankton_yolo26n',
        workers=workers,
        amp=amp
    )
    
    print("\nTraining completed!")
    print(f"Results saved to: runs/detect/zooplankton_yolo26n")
    
    # Validate
    print("\nValidating model...")
    metrics = detector.validate(data_yaml)
    print(f"Validation metrics:")
    print(f"  mAP@0.5: {metrics['map50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['map']:.4f}")
    
    return detector


class AlbumentationsDataset(Dataset):
    """Wraps torchvision ImageFolder to apply Albumentations transforms instead."""
    def __init__(self, root, transform=None, binary_mapping=None, class_to_idx=None):
        self.image_folder = datasets.ImageFolder(root)
        # If class_to_idx is provided (e.g. from the train set), override
        # ImageFolder's auto-detected mapping so val/test labels align.
        if class_to_idx is not None:
            self.image_folder.class_to_idx = class_to_idx
            # Re-map all samples to use the provided class_to_idx
            self.image_folder.samples = [
                (path, class_to_idx[self.image_folder.classes[label]])
                for path, label in self.image_folder.samples
                if self.image_folder.classes[label] in class_to_idx
            ]
            self.image_folder.targets = [s[1] for s in self.image_folder.samples]
        self.transform = transform
        self.binary_mapping = binary_mapping # Dict[int, int] (N classes -> 0 or 1)

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        path, label = self.image_folder.samples[idx]
        image = cv2.imread(path)
        if image is None:
            # Fallback for corrupted images
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        if self.binary_mapping is not None:
            label = self.binary_mapping[label]

        return image, label

def train_binary_filter(data_dir: str, epochs: int = 10, batch_size: int = 64, device: str = 'cuda', workers: int = 8):
    print("=" * 50)
    print("Training Two-Stage Binary Filter (Plankton vs Artifact)")
    print("=" * 50)
    
    # Load hierarchy to create binary mapping
    data_path = Path(data_dir)
    mapping_file = data_path.parent / 'hierarchy.json'
    class_map_file = data_path.parent / 'class_mapping.json'
    
    binary_mapping = {}
    if mapping_file.exists() and class_map_file.exists():
        with open(mapping_file, 'r') as f:
            hier = json.load(f)['class_to_higher']

        # ImageFolder assigns integer labels based on sorted folder names ON DISK.
        # We must use the actual folders (which include 'rare_species' and exclude
        # merged classes) — NOT the class_mapping.json which has the original 127.
        train_folders = sorted([
            d.name for d in (data_path / 'train').iterdir() if d.is_dir()
        ])
        for idx, folder_name in enumerate(train_folders):
            higher = hier.get(folder_name, 'plankton')
            # 0 for artifact, 1 for plankton
            binary_mapping[idx] = 0 if higher == 'artifact' else 1
    else:
        print("Warning: hierarchy.json not found, skipping binary filter training.")
        return None

    train_transform = A.Compose([
        A.Resize(128, 128),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    train_dataset = AlbumentationsDataset(data_path / 'train', train_transform, binary_mapping)
    val_dataset = AlbumentationsDataset(data_path / 'val', val_transform, binary_mapping,
                                        class_to_idx=train_dataset.image_folder.class_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    model = TwoStageBinaryFilter(input_size=128, device=device)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    save_path = 'models/binary_filter_best.pt'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    model.model.train()
    for epoch in range(epochs):
        train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
        val_loss /= len(val_loader)
        acc = 100 * correct / total
        print(f"Binary Filter Epoch {epoch+1}/{epochs} | Val Loss: {val_loss:.4f} | Acc: {acc:.1f}%")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.model.state_dict(), save_path)
            
        model.model.train()
        
    print(f"Binary Filter training complete. Saved to {save_path}")
    return model

def train_classifier(data_dir: str,
                    num_classes: int,
                    epochs: int = 50,
                    batch_size: int = 32,
                    learning_rate: float = 0.001,
                    device: str = 'cuda',
                    workers: int = 8,
                    amp: bool = True,
                    embed_preprocessing: bool = True):
    """
    Train MobileNetV4 Main Classifier.
    """
    print("=" * 50)
    print("Training MobileNet Main Classifier")
    print("=" * 50)
    
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = 'cpu'

    # Using robust Albumentations transforms
    if embed_preprocessing:
        # Preprocessing (CLAHE approx + normalization) is done inside the model.
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            A.GaussNoise(std_range=(0.04, 0.22), p=0.2),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ])
    else:
        # Legacy path: full transforms including ImageNet normalization.
        train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    # Load datasets
    print("\nLoading datasets...")
    data_path = Path(data_dir)
    
    train_dataset = AlbumentationsDataset(data_path / 'train', transform=train_transform)
    val_dataset = AlbumentationsDataset(data_path / 'val', transform=val_transform,
                                        class_to_idx=train_dataset.image_folder.class_to_idx)
    
    # Auto-detect actual class count from disk (may differ from --num-classes
    # after rare class merging in prepare_dataset.py)
    actual_num_classes = len(train_dataset.image_folder.classes)
    if actual_num_classes != num_classes:
        print(f"  WARNING: --num-classes={num_classes} but found {actual_num_classes} folders on disk. Using {actual_num_classes}.")
        num_classes = actual_num_classes
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    
    # Initialize classifier
    classifier = MobileNetClassifier(
        num_classes=num_classes,
        device=device,
        use_embeddings=True,
        embed_preprocessing=embed_preprocessing,
    )
    
    # Print model info
    info = classifier.get_model_info()
    print(f"\nModel Information:")
    print(f"  Type: {info['model_type']}")
    print(f"  Classes: {info['num_classes']}")
    print(f"  Parameters: {info['total_parameters']:,}")
    print(f"  Model size: {info['model_size_mb']:.2f} MB")
    
    # Train
    print(f"\nStarting training...")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    save_path = 'models/mobilenetv3_classifier_best.pt'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        save_path=save_path,
        use_amp=amp
    )
    
    print(f"\nTraining completed!")
    print(f"Best model saved to: {save_path}")
    
    return classifier


def export_models(detector: YOLODetector, 
                 classifier: MobileNetClassifier,
                 output_dir: str = 'models/tflite'):
    """
    Export models to TFLite for edge deployment
    
    Args:
        detector: Trained detector
        classifier: Trained classifier
        output_dir: Output directory
    """
    print("\n" + "=" * 50)
    print("Exporting Models to TFLite")
    print("=" * 50)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export detector
    print("\nExporting detector...")
    detector.export_to_tflite(
        str(output_path / 'yolov5n_detector.tflite'),
        quantize=True
    )
    
    # Export classifier
    print("\nExporting classifier...")
    classifier.export_to_tflite(
        str(output_path / 'mobilenetv3_classifier.tflite'),
        quantize=True
    )
    
    print(f"\nModels exported to: {output_path}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Zooplankton models')
    parser.add_argument('--mode', type=str, choices=['detector', 'classifier', 'both'],
                       default='both', help='Which model to train')
    parser.add_argument('--data-yaml', type=str, 
                       default='prepared_dataset/yolo/data.yaml',
                       help='Path to YOLO data.yaml')
    parser.add_argument('--classification-dir', type=str,
                       default='prepared_dataset/classification',
                       help='Path to classification dataset')
    parser.add_argument('--num-classes', type=int, default=127,
                       help='Number of classes')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of dataloader workers')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable Automatic Mixed Precision')
    parser.add_argument('--export', action='store_true',
                       help='Export models to ONNX after training')
    parser.add_argument('--no-embed-preprocessing', action='store_true',
                       help='Disable in-model ZooScanPreprocessor (legacy behaviour)')
    parser.add_argument('--prune-ratio', type=float, default=0.0,
                       help='Fraction of classifier weights to prune after training '
                            '(0.0 = disabled, e.g. 0.3 = 30%% sparsity)')
    parser.add_argument('--prune-structured', action='store_true',
                       help='Use structured (channel) pruning instead of global L1 '
                            'unstructured pruning')
    parser.add_argument('--prune-epochs', type=int, default=5,
                       help='Fine-tuning epochs after pruning to recover accuracy')
    parser.add_argument('--prune-only', action='store_true',
                       help='Skip all training — load saved weights and run pruning + fine-tune only. '
                            'Requires models/mobilenetv3_classifier_best.pt to exist.')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    print("=" * 50)
    print("Zooplankton Model Training")
    print("=" * 50)
    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {args.device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    
    detector = None
    classifier = None
    embed_preprocessing = not args.no_embed_preprocessing
    
    # ── Prune-only mode: skip all training, load saved weights ─────────────
    if args.prune_only:
        print("\n" + "=" * 50)
        print("PRUNE-ONLY MODE — Loading saved classifier weights")
        print("=" * 50)
        
        # Auto-detect num_classes from disk to avoid arg mismatch
        cls_train_dir = Path(args.classification_dir) / 'train'
        if cls_train_dir.exists():
            n_on_disk = len([d for d in cls_train_dir.iterdir() if d.is_dir()])
            if n_on_disk != args.num_classes:
                print(f"  WARNING: --num-classes={args.num_classes} but found {n_on_disk} folders on disk. Using {n_on_disk}.")
                args.num_classes = n_on_disk

        saved_weights = 'models/mobilenetv3_classifier_best.pt'
        if not Path(saved_weights).exists():
            print(f"ERROR: {saved_weights} not found. Run the full training job first.")
            import sys; sys.exit(1)
        
        print(f"  Loading weights from {saved_weights} ...")
        classifier = MobileNetClassifier(
            model_path=saved_weights,
            num_classes=args.num_classes,
            device=args.device,
            embed_preprocessing=embed_preprocessing,
        )
        info = classifier.get_model_info()
        print(f"  Model: {info['model_type']}  Classes: {info['num_classes']}  Params: {info['total_parameters']:,}")
        
        if args.prune_ratio <= 0.0:
            print("ERROR: --prune-only requires --prune-ratio > 0.0")
            import sys; sys.exit(1)

    else:
        # ── Normal training path ──────────────────────────────────────────────
        # Train detector
        if args.mode in ['detector', 'both']:
            detector = train_detector(
                data_yaml=args.data_yaml,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                workers=args.workers,
                amp=not args.no_amp
            )
        
        # Train binary filter first
        if args.mode in ['classifier', 'both']:
            binary_filter = train_binary_filter(
                data_dir=args.classification_dir,
                epochs=10,
                batch_size=args.batch_size * 2, # Faster throughput
                device=args.device,
                workers=args.workers
            )
            
        # Train classifier
        if args.mode in ['classifier', 'both']:
            classifier = train_classifier(
                data_dir=args.classification_dir,
                num_classes=args.num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                workers=args.workers,
                amp=not args.no_amp,
                embed_preprocessing=embed_preprocessing,
            )

    # ── Post-training pruning ───────────────────────────────────────────────
    if classifier is not None and args.prune_ratio > 0.0:
        print("\n" + "=" * 50)
        print(f"Pruning classifier (ratio={args.prune_ratio}, "
              f"structured={args.prune_structured})")
        print("=" * 50)

        classifier.prune(
            sparsity=args.prune_ratio,
            structured=args.prune_structured,
            make_permanent=True,
        )

        if args.prune_epochs > 0:
            # Reload data loaders for fine-tuning
            from torch.utils.data import DataLoader
            from torchvision import datasets

            data_path = Path(args.classification_dir)
            fine_transform = A.Compose([
                A.Resize(224, 224),
                A.HorizontalFlip(p=0.5),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ])
            val_transform = A.Compose([
                A.Resize(224, 224),
                A.ToFloat(max_value=255.0),
                ToTensorV2(),
            ])

            fine_train_ds = AlbumentationsDataset(data_path / 'train', fine_transform)
            fine_train = DataLoader(
                fine_train_ds,
                batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True,
            )
            fine_val = DataLoader(
                AlbumentationsDataset(data_path / 'val', val_transform,
                                      class_to_idx=fine_train_ds.image_folder.class_to_idx),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )

            pruned_save = 'models/mobilenetv3_classifier_pruned.pt'
            classifier.fine_tune_pruned(
                train_loader=fine_train,
                val_loader=fine_val,
                epochs=args.prune_epochs,
                save_path=pruned_save,
            )
            print(f"Pruned model saved → {pruned_save}")

        # Print final sparsity report
        stats = classifier.get_sparsity()
        print(f"Final sparsity: {stats['achieved_sparsity']:.1%} "
              f"({stats['zero_params']:,} / {stats['total_params']:,} weights)")
    
    # Export
    if args.export and detector is not None and classifier is not None:
        print("\n" + "=" * 50)
        print("Exporting models to ONNX")
        print("=" * 50)
        Path('models/onnx').mkdir(parents=True, exist_ok=True)
        classifier.export_to_onnx('models/onnx/mobilenetv3_classifier.onnx')
        print("Note: export_to_tflite for detector requires ultralytics export()")
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    main()
