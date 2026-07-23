"""
MobileNetV3 Classifier Module
Lightweight species classification for zooplankton.

Key additions over the baseline:
  - ZooScanPreprocessor: differentiable CLAHE approximation + normalization
    embedded as the first nn.Module of the model, so raw [0,1] images can
    be fed at inference time with no external OpenCV preprocessing.
  - prune(): L1 global unstructured pruning (or per-layer structured) to
    reduce weight count before edge deployment.
  - export_full_model(): exports preprocessor + backbone as one ONNX graph.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune_utils
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torchvision import models, transforms
from PIL import Image
import json

# ---------------------------------------------------------------------------
# Focal Loss for Class Imbalance
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for handling extreme class imbalance (e.g. rare species vs detritus).
    Down-weights well-classified examples to focus on hard examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss



# ---------------------------------------------------------------------------
# Differentiable preprocessing head
# ---------------------------------------------------------------------------

class ZooScanPreprocessor(nn.Module):
    """
    Differentiable in-model preprocessing head for ZooScan microscopy images.

    Approximates the offline OpenCV preprocessing as pure PyTorch ops so the
    full pipeline (preprocessing → feature extraction → classification) can be
    exported as a single ONNX/TFLite graph and fine-tuned end-to-end.

    Operations applied in forward():
      1. Local Contrast Normalization  — differentiable CLAHE approximation.
         Uses AvgPool2d to estimate local mean and variance per tile, then
         normalises and clips std to mirror CLAHE's clip-limit behaviour.
      2. Learnable gamma correction    — single scalar γ (log-parameterised,
         init = 1.0 identity). Adapts to dataset brightness distribution
         during fine-tuning.
      3. ImageNet normalisation        — fixed mean/std registered as buffers
         (not trained); required for the pretrained MobileNetV3 backbone.

    Input contract:
      FloatTensor  [B, 3, H, W]  in range [0.0, 1.0]
      (i.e. divide uint8 pixels by 255 before calling the model)
    """

    def __init__(
        self,
        lcn_kernel: int = 28,
        # tile size: CLAHE uses tileGridSize=8 on 224px → 224/8 = 28 px/tile
        clip_limit: float = 0.03,
        # normalised clip limit: CLAHE clip=2.0 on 8×8 hist → 2/(8*8)≈0.03
        imagenet_normalize: bool = True,
    ):
        super().__init__()
        self.clip_limit = clip_limit
        self.imagenet_normalize = imagenet_normalize

        # Local statistics estimators (non-trainable fixed pooling)
        # Use odd kernel (27) so same-padding = (27-1)//2 = 13 keeps output = input size
        self.lcn_kernel = 27  # closest odd number to 28 for CLAHE tile
        pad = (self.lcn_kernel - 1) // 2
        self.local_avg = nn.AvgPool2d(
            kernel_size=self.lcn_kernel, stride=1, padding=pad, count_include_pad=False
        )

        # Learnable log-gamma (exp ensures γ > 0; init 0 → γ = 1.0 identity)
        self.log_gamma = nn.Parameter(torch.zeros(1))

        # ImageNet normalisation stats — fixed buffers (not parameters)
        self.register_buffer(
            "norm_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "norm_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── 1. Local Contrast Normalization ─────────────────────────────────
        local_mean = self.local_avg(x)
        local_sq_mean = self.local_avg(x * x)
        local_var = (local_sq_mean - local_mean * local_mean).clamp(min=0.0)
        local_std = local_var.sqrt().clamp(min=self.clip_limit)
        x = (x - local_mean) / local_std
        # Squash back to [0, 1] — sigmoid is smooth and differentiable
        x = torch.sigmoid(x)

        # ── 2. Learnable gamma correction ────────────────────────────────────
        gamma = self.log_gamma.exp().clamp(0.5, 2.0)
        x = x.pow(gamma)

        # ── 3. ImageNet normalisation ────────────────────────────────────────
        if self.imagenet_normalize:
            x = (x - self.norm_mean) / self.norm_std

        return x


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MobileNetClassifier:
    """
    MobileNetV3-Small classifier for zooplankton species identification.

    When embed_preprocessing=True (default), a ZooScanPreprocessor is stored
    as self.preprocessor and applied inside classify() / extract_embeddings().
    The inference transform then only resizes and converts to tensor — no
    external CLAHE or normalization step is needed.

    Use self.full_model for ONNX/TFLite export (preprocessor + backbone fused).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 127,
        input_size: int = 224,
        device: str = "cpu",
        use_embeddings: bool = True,
        embed_preprocessing: bool = True,
    ):
        """
        Args:
            model_path:          Path to trained model weights (.pt state dict).
            num_classes:         Number of species/genus classes.
            input_size:          Square side-length for input images (px).
            device:              Torch device string ('cpu', 'cuda').
            use_embeddings:      Return embeddings from the penultimate layer
                                 in extract_embeddings().
            embed_preprocessing: Attach ZooScanPreprocessor to the model.
                                 When True, raw [0,1] images are accepted at
                                 inference time — no external CLAHE needed.
        """
        self.num_classes = num_classes
        self.input_side = input_size       # scalar for resize
        self.device = device
        self.use_embeddings = use_embeddings
        self.embed_preprocessing = embed_preprocessing
        self.model_type = "Unknown"

        # Build backbone
        self.model = self._build_model()

        # Preprocessing head (registered here, not inside backbone so that
        # self.model.features / avgpool / classifier are still accessible)
        self.preprocessor: Optional[ZooScanPreprocessor] = (
            ZooScanPreprocessor() if embed_preprocessing else None
        )

        # Load weights if provided
        if model_path is not None and Path(model_path).exists():
            self._load_weights(model_path)

        self.model.to(device)
        self.model.eval()
        if self.preprocessor is not None:
            self.preprocessor.to(device)
            self.preprocessor.eval()

        # Inference transform:
        #   - embed_preprocessing=True  → just resize + to_tensor
        #     (model preprocessor handles CLAHE + normalization)
        #   - embed_preprocessing=False → resize + CLAHE + normalize
        self.transform = self._build_transform(train=False)

        # input_size kept as tuple for backward-compat with get_model_info()
        self.input_size = (input_size, input_size, 3)

        # Class names mapping
        self.class_names: Dict[int, str] = {}

    # ── Construction helpers ────────────────────────────────────────────────

    def _build_model(self) -> nn.Module:
        """Build MobileNetV4 (via timm) or fallback to MobileNetV3-Small."""
        try:
            import timm
            # Use MobileNetV4 for pure CPU optimization
            model = timm.create_model(
                'mobilenetv4_conv_small.e2400_r224_in1k', 
                pretrained=True, 
                num_classes=self.num_classes
            )
            self.model_type = "MobileNetV4-Conv-Small"
            return model
        except Exception as e:
            print(f"timm / MobileNetV4 load failed ({e}), falling back to MobileNetV3-Small")
            self.model_type = "MobileNetV3-Small"
            model = models.mobilenet_v3_small(weights="DEFAULT")
            model.classifier = nn.Sequential(
                nn.Linear(576, 1024),
                nn.Hardswish(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, self.num_classes),
            )
            return model

    def _build_transform(self, train: bool = False) -> transforms.Compose:
        """
        Build the image transform applied before the model.

        When embed_preprocessing is True, CLAHE and normalization are handled
        by the in-model ZooScanPreprocessor, so they are omitted here.
        """
        steps = [transforms.ToPILImage()]

        if train:
            steps += [
                transforms.Resize((self.input_side, self.input_side)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        else:
            steps.append(transforms.Resize((self.input_side, self.input_side)))

        steps.append(transforms.ToTensor())  # → [0, 1] float

        if not self.embed_preprocessing:
            # Legacy path: apply CLAHE-like contrast + ImageNet normalisation
            # outside the model (original behaviour).
            steps += [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                )
            ]

        return transforms.Compose(steps)

    def _load_weights(self, model_path: str):
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights: {e}")

    # ── Inference ───────────────────────────────────────────────────────────

    def _preprocess_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply preprocessor (if embedded) to a batch tensor."""
        if self.preprocessor is not None:
            return self.preprocessor(tensor)
        return tensor

    def classify(self, image: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Classify a single image.

        Args:
            image: BGR or grayscale uint8 numpy array.
            top_k: Number of top predictions to return.

        Returns:
            List of (class_id, confidence) tuples ordered by confidence desc.
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        tensor = self._preprocess_tensor(tensor)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probs, top_k)
        return [(int(i.cpu()), float(p.cpu())) for p, i in zip(top_probs[0], top_indices[0])]

    def classify_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 16,
        top_k: int = 5,
    ) -> List[List[Tuple[int, float]]]:
        """Classify a list of images in batches."""
        all_predictions = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_tensors = []
            for img in batch:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_tensors.append(self.transform(img_rgb))

            batch_tensor = torch.stack(batch_tensors).to(self.device)
            batch_tensor = self._preprocess_tensor(batch_tensor)

            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = F.softmax(outputs, dim=1)

            top_probs, top_indices = torch.topk(probs, top_k)
            for j in range(len(batch)):
                preds = [
                    (int(top_indices[j][k].cpu()), float(top_probs[j][k].cpu()))
                    for k in range(top_k)
                ]
                all_predictions.append(preds)

        return all_predictions

    # ── Embedding extraction ────────────────────────────────────────────────

    def extract_embeddings(self, image: np.ndarray) -> np.ndarray:
        """Extract feature embeddings from the penultimate layer."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        tensor = self._preprocess_tensor(tensor)

        with torch.no_grad():
            if "MobileNetV4" in self.model_type:
                embeddings = self.model.forward_head(self.model.forward_features(tensor), pre_logits=True)
            else:
                features = self.model.features(tensor)
                features = self.model.avgpool(features)
                embeddings = torch.flatten(features, 1)
                if self.use_embeddings:
                    embeddings = self.model.classifier[:3](embeddings)  # up to Dropout

        return embeddings.cpu().numpy().flatten()

    def extract_embeddings_batch(
        self, images: List[np.ndarray], batch_size: int = 16
    ) -> np.ndarray:
        """Extract embeddings for a list of images."""
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            batch_tensors = []
            for img in batch:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_tensors.append(self.transform(img_rgb))

            batch_tensor = torch.stack(batch_tensors).to(self.device)
            batch_tensor = self._preprocess_tensor(batch_tensor)

            with torch.no_grad():
                if "MobileNetV4" in self.model_type:
                    embeddings = self.model.forward_head(self.model.forward_features(batch_tensor), pre_logits=True)
                else:
                    features = self.model.features(batch_tensor)
                    features = self.model.avgpool(features)
                    embeddings = torch.flatten(features, 1)
                    if self.use_embeddings:
                        embeddings = self.model.classifier[:3](embeddings)

            all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    # ── Class name helpers ──────────────────────────────────────────────────

    def set_class_names(self, class_names: Dict[int, str]):
        self.class_names = class_names

    def get_class_name(self, class_id: int) -> str:
        return self.class_names.get(class_id, f"Class_{class_id}")

    def classify_with_names(
        self, image: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        return [
            (self.get_class_name(cid), conf)
            for cid, conf in self.classify(image, top_k)
        ]

    def classify_hierarchical(self, image: np.ndarray, hierarchy_mapping: Dict[str, str], top_k: int = 5) -> Dict[str, List[Tuple[str, float]]]:
        """
        Classifies the image and aggregates probabilities by higher-level taxa 
        (e.g. summing Copepoda species probabilities to get a Copepoda probability).
        Provides a safe fallback for edge deployment if species confidence is low.
        """
        if not self.class_names:
            raise ValueError("class_names mapping not set")
            
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        tensor = self._preprocess_tensor(tensor)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)[0].cpu().numpy()

        # Species level
        species_probs = [(self.get_class_name(i), float(p)) for i, p in enumerate(probs)]
        species_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Higher level
        higher_probs = {}
        for i, p in enumerate(probs):
            cls_name = self.get_class_name(i)
            higher_name = hierarchy_mapping.get(cls_name, "Unknown")
            higher_probs[higher_name] = higher_probs.get(higher_name, 0.0) + float(p)
            
        higher_probs_list = list(higher_probs.items())
        higher_probs_list.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "species": species_probs[:top_k],
            "higher_taxa": higher_probs_list[:top_k]
        }

    # ── Pruning ─────────────────────────────────────────────────────────────

    def prune(
        self,
        sparsity: float = 0.3,
        structured: bool = False,
        make_permanent: bool = True,
    ) -> Dict:
        """
        Prune model weights to reduce computation for edge deployment.

        Two modes:
          structured=False  Global L1 unstructured pruning.
                            Zeroes out the lowest-magnitude weights across all
                            layers simultaneously.  Sparsity is exact.
                            Compressed representation requires sparse-aware
                            runtime (e.g. XNNPACK on Raspberry Pi).

          structured=True   Per-layer structured (filter) pruning.
                            Removes entire output filters with the lowest L1
                            norm, physically reducing tensor dimensions.
                            Reduces FLOPs and model size without sparse runtime.
                            Note: depthwise conv groups are left untouched to
                            avoid shape mismatches in MobileNetV3.

        Call fine_tune_pruned() after this to recover accuracy.

        Args:
            sparsity:        Fraction of weights to remove (0.0 – 1.0).
            structured:      Use structured (channel) pruning if True.
            make_permanent:  Remove masks and bake zeros into weights.
                             Required before ONNX/TFLite export.

        Returns:
            Dict with sparsity statistics.
        """
        self.model.train()  # prune needs grad-enabled mode internally

        # Collect prunable layers — skip the final classification layer so
        # output shape is preserved.
        if "MobileNetV4" in self.model_type:
            # timm models: get_classifier() returns the final Linear layer
            final_linear = self.model.get_classifier()
        else:
            final_linear = self.model.classifier[-1]
        params_to_prune = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module is final_linear:
                    continue
                # Skip depthwise convs in structured mode (groups != 1)
                if structured and isinstance(module, nn.Conv2d) and module.groups > 1:
                    continue
                params_to_prune.append((module, "weight"))

        if not params_to_prune:
            print("Warning: No prunable layers found.")
            return {}

        if structured:
            for module, _ in params_to_prune:
                if isinstance(module, nn.Conv2d):
                    prune_utils.ln_structured(
                        module, name="weight", amount=sparsity, n=1, dim=0
                    )
                else:
                    prune_utils.l1_unstructured(module, name="weight", amount=sparsity)
        else:
            prune_utils.global_unstructured(
                params_to_prune,
                pruning_method=prune_utils.L1Unstructured,
                amount=sparsity,
            )

        if make_permanent:
            for module, param in params_to_prune:
                try:
                    prune_utils.remove(module, param)
                except ValueError:
                    pass

        self.model.eval()

        # Compute achieved sparsity
        stats = self.get_sparsity()
        stats["requested_sparsity"] = sparsity
        stats["structured"] = structured

        print(
            f"Pruning complete  ({stats['achieved_sparsity']:.1%} sparsity, "
            f"mode={'structured' if structured else 'unstructured'})"
        )
        print(
            f"  {stats['zero_params']:,} / {stats['total_params']:,} "
            "backbone weights zeroed"
        )
        return stats

    def get_sparsity(self) -> Dict:
        """Return current weight sparsity statistics for the backbone."""
        total, zeros = 0, 0
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w = module.weight
                total += w.numel()
                zeros += int((w == 0).sum())
        sparsity = zeros / total if total > 0 else 0.0
        return {
            "total_params": total,
            "zero_params": zeros,
            "achieved_sparsity": sparsity,
        }

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        learning_rate: float = 0.001,
        save_path: Optional[str] = None,
        use_amp: bool = True,
    ):
        """
        Train the classifier.

        When embed_preprocessing=True, the preprocessor's learnable gamma
        parameter is included in the optimiser so it adapts to the dataset.
        The DataLoader transforms should NOT apply CLAHE or ImageNet
        normalization when embed_preprocessing=True (the model does it).
        See scripts/train_models.py for the correct transform setup.
        """
        # Collect parameters: backbone + preprocessor (if embedded)
        params = list(self.model.parameters())
        if self.preprocessor is not None:
            params += list(self.preprocessor.parameters())

        self.model.train()
        if self.preprocessor is not None:
            self.preprocessor.train()

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True

        criterion = FocalLoss(gamma=2.0) # Replaced CrossEntropyLoss with FocalLoss
        optimizer = torch.optim.Adam(params, lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=use_amp and self.device == "cuda"
        )

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # ── Training phase ──────────────────────────────────────────────
            train_loss, train_correct, train_total = 0.0, 0, 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Apply embedded preprocessor during training
                if self.preprocessor is not None:
                    inputs = self.preprocessor(inputs)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast(
                    enabled=use_amp and self.device == "cuda"
                ):
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total

            # ── Validation phase ────────────────────────────────────────────
            self.model.eval()
            if self.preprocessor is not None:
                self.preprocessor.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    if self.preprocessor is not None:
                        inputs = self.preprocessor(inputs)
                    with torch.cuda.amp.autocast(
                        enabled=use_amp and self.device == "cuda"
                    ):
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = 100 * val_correct / val_total
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{epochs}  "
                f"train_loss={train_loss:.4f} acc={train_acc:.1f}%  "
                f"val_loss={val_loss:.4f} acc={val_acc:.1f}%"
            )
            if self.preprocessor is not None:
                gamma = self.preprocessor.log_gamma.exp().item()
                print(f"  preprocessor γ = {gamma:.4f}")

            if save_path and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved best model → {save_path}")

            self.model.train()
            if self.preprocessor is not None:
                self.preprocessor.train()

        if save_path and Path(save_path).exists():
            self._load_weights(save_path)

        self.model.eval()
        if self.preprocessor is not None:
            self.preprocessor.eval()

    def fine_tune_pruned(
        self,
        train_loader,
        val_loader,
        epochs: int = 5,
        learning_rate: float = 0.0001,
        save_path: Optional[str] = None,
    ):
        """
        Short fine-tuning pass to recover accuracy after pruning.

        Uses a low learning rate to avoid disturbing the pruning structure.
        """
        print(f"\nFine-tuning pruned model for {epochs} epochs (lr={learning_rate})...")
        self.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            save_path=save_path,
            use_amp=True,
        )

    # ── Export ──────────────────────────────────────────────────────────────

    @property
    def full_model(self) -> nn.Module:
        """
        Combined preprocessor + backbone as a single nn.Sequential.

        Use this for export:
          torch.onnx.export(classifier.full_model, ...)
        The exported graph accepts raw [0,1] float images and returns logits.
        """
        if self.preprocessor is not None:
            return nn.Sequential(
                OrderedDict([
                    ("preprocessor", self.preprocessor),
                    ("backbone", self.model),
                ])
            )
        return self.model

    def export_to_onnx(self, output_path: str):
        """
        Export the full model (preprocessor + backbone) to ONNX.

        The ONNX graph accepts [1, 3, H, W] float32 in [0, 1].
        Convert to TFLite via: onnx-tf / tf2onnx on the target platform.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        input_h, input_w = self.input_size[0], self.input_size[1]
        dummy = torch.randn(1, 3, input_h, input_w).to(self.device)

        full = self.full_model.eval()
        torch.onnx.export(
            full,
            dummy,
            str(output_path),
            export_params=True,
            opset_version=11,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}},
        )
        print(f"Exported ONNX → {output_path}")

    def export_to_tflite(self, output_path: str, quantize: bool = True):
        """Export via ONNX (see export_to_onnx). TFLite conversion is external."""
        onnx_path = Path(output_path).with_suffix(".onnx")
        self.export_to_onnx(str(onnx_path))
        print("Convert to TFLite with: tf2onnx or onnx-tensorflow")

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def get_model_info(self) -> Dict:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        prep_params = (
            sum(p.numel() for p in self.preprocessor.parameters())
            if self.preprocessor is not None
            else 0
        )
        sparsity = self.get_sparsity() if "MobileNetV3" in self.model_type else {"achieved_sparsity": 0.0}
        return {
            "model_type": self.model_type,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "device": self.device,
            "embed_preprocessing": self.embed_preprocessing,
            "preprocessor_params": prep_params,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "model_size_mb": total * 4 / (1024 ** 2),
            "backbone_sparsity": f"{sparsity['achieved_sparsity']:.1%}",
        }

# ---------------------------------------------------------------------------
# Two-Stage Binary Filter
# ---------------------------------------------------------------------------
class TwoStageBinaryFilter:
    """
    Ultra-lightweight binary classifier (Plankton vs Artifact) to execute 
    BEFORE the heavy 127-class MobileNet. Maximizes edge CPU FPS by dropping garbage.
    """
    def __init__(self, model_path: Optional[str] = None, input_size: int = 128, device: str = "cpu"):
        self.device = device
        self.input_size = input_size
        
        # Tiny backbone for maximum FPS
        self.model = models.mobilenet_v3_small(weights="DEFAULT")
        self.model.classifier = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2) # 0 = Artifact/Noise, 1 = Plankton
        )
        self.model.to(device)
        self.model.eval()

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def is_plankton(self, image: np.ndarray, threshold: float = 0.5) -> bool:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)
        
        # Return True if class 1 (Plankton) probability > threshold
        return float(probs[0][1].cpu()) > threshold
