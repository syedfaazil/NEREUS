"""
Standalone pruning + fine-tune script.
Loads an existing trained classifier and runs pruning + fine-tuning.
Called by submit_job_2.pbs — no changes to train_models.py needed.
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from classification import MobileNetClassifier


class AlbumentationsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_folder = datasets.ImageFolder(root)
        self.transform = transform

    def __len__(self):
        return len(self.image_folder)

    def __getitem__(self, idx):
        path, label = self.image_folder.samples[idx]
        image = cv2.imread(path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label


def main():
    parser = argparse.ArgumentParser(description='Prune + fine-tune a saved classifier')
    parser.add_argument('--classification-dir', required=True,
                        help='Path to classification dataset (contains train/ val/)')
    parser.add_argument('--weights', default='models/mobilenetv3_classifier_best.pt',
                        help='Path to saved classifier weights')
    parser.add_argument('--num-classes', type=int, default=0,
                        help='Number of classes (0 = auto-detect from disk)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prune-ratio', type=float, default=0.3,
                        help='Fraction of weights to prune (0.0–1.0)')
    parser.add_argument('--prune-structured', action='store_true',
                        help='Use structured (channel) pruning')
    parser.add_argument('--prune-epochs', type=int, default=10,
                        help='Fine-tuning epochs after pruning')
    args = parser.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    # ── Auto-detect num_classes ───────────────────────────────────────────────
    cls_dir = Path(args.classification_dir)
    train_dir = cls_dir / 'train'
    if args.num_classes == 0:
        args.num_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
        print(f"Auto-detected {args.num_classes} classes from {train_dir}")

    # ── Load saved weights ────────────────────────────────────────────────────
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: Weights not found at {weights_path}")
        sys.exit(1)

    print("=" * 50)
    print("PRUNE-ONLY MODE — Loading saved classifier weights")
    print("=" * 50)
    print(f"  Weights : {weights_path}")
    print(f"  Classes : {args.num_classes}")
    print(f"  Device  : {args.device}")

    classifier = MobileNetClassifier(
        model_path=str(weights_path),
        num_classes=args.num_classes,
        device=args.device,
        embed_preprocessing=True,
    )
    info = classifier.get_model_info()
    print(f"  Model   : {info['model_type']}  Params: {info['total_parameters']:,}")

    # ── Prune ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Pruning  ratio={args.prune_ratio}  structured={args.prune_structured}")
    print("=" * 50)

    classifier.prune(
        sparsity=args.prune_ratio,
        structured=args.prune_structured,
        make_permanent=True,
    )

    # ── Fine-tune ──────────────────────────────────────────────────────────────
    if args.prune_epochs > 0:
        print("\n" + "=" * 50)
        print(f"Fine-tuning for {args.prune_epochs} epochs after pruning")
        print("=" * 50)

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

        fine_train = DataLoader(
            AlbumentationsDataset(train_dir, fine_transform),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True,
        )
        fine_val = DataLoader(
            AlbumentationsDataset(cls_dir / 'val', val_transform),
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

    # ── Final sparsity report ──────────────────────────────────────────────────
    stats = classifier.get_sparsity()
    print(f"\nFinal sparsity: {stats['achieved_sparsity']:.1%} "
          f"({stats['zero_params']:,} / {stats['total_params']:,} weights zeroed)")
    print("\nPruning + fine-tune complete!")


if __name__ == '__main__':
    main()
