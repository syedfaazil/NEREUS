"""
YOLO26n Detector Module
Lightweight object detection for zooplankton specimens
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from ultralytics import YOLO


class YOLODetector:
    """
    YOLO26n-based detector for zooplankton specimens
    Optimized for Raspberry Pi deployment
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 device: str = 'cpu'):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to trained model weights. None to use pretrained
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on ('cpu', 'cuda', 'mps')
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Load model
        if model_path is not None and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Use YOLO26n pretrained model as starting point
            self.model = YOLO('yolo26n.pt')
        
        # Set device
        self.model.to(device)
        
        # Model info
        self.input_size = 640  # Default YOLOv5 input size
    
    def detect(self, image: np.ndarray, 
               return_crops: bool = False) -> Tuple[List[Dict], Optional[List[np.ndarray]]]:
        """
        Detect objects in image
        
        Args:
            image: Input image (BGR or grayscale)
            return_crops: Whether to return cropped detections
            
        Returns:
            List of detection dictionaries and optionally cropped images
        """
        # Ensure image is in correct format
        if len(image.shape) == 2:
            # Convert grayscale to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Parse results
        detections = []
        crops = [] if return_crops else None
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                # Get box coordinates
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)
                
                # Get confidence and class
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'class': cls,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1)
                }
                
                detections.append(detection)
                
                # Extract crop if requested
                if return_crops:
                    crop = image[y1:y2, x1:x2].copy()
                    crops.append(crop)
        
        return detections, crops
    
    def detect_batch(self, images: List[np.ndarray], 
                    batch_size: int = 8) -> List[List[Dict]]:
        """
        Detect objects in batch of images
        
        Args:
            images: List of input images
            batch_size: Batch size for inference
            
        Returns:
            List of detection lists for each image
        """
        all_detections = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Ensure all images are BGR
            processed_batch = []
            for img in batch:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                processed_batch.append(img)
            
            # Run inference
            results = self.model.predict(
                processed_batch,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Parse results for each image
            for result in results:
                detections = []
                boxes = result.boxes
                
                for j in range(len(boxes)):
                    box = boxes.xyxy[j].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    
                    conf = float(boxes.conf[j].cpu().numpy())
                    cls = int(boxes.cls[j].cpu().numpy())
                    
                    detection = {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf,
                        'class': cls,
                        'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                        'width': x2 - x1,
                        'height': y2 - y1,
                        'area': (x2 - x1) * (y2 - y1)
                    }
                    
                    detections.append(detection)
                
                all_detections.append(detections)
        
        return all_detections
    
    def visualize_detections(self, image: np.ndarray, 
                            detections: List[Dict],
                            class_names: Optional[Dict[int, str]] = None) -> np.ndarray:
        """
        Visualize detections on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            class_names: Dictionary mapping class IDs to names
            
        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()
        
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls = det['class']
            
            # Draw bounding box
            color = self._get_color(cls)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            if class_names is not None and cls in class_names:
                label = f"{class_names[cls]}: {conf:.2f}"
            else:
                label = f"Class {cls}: {conf:.2f}"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        Get color for class ID
        
        Args:
            class_id: Class ID
            
        Returns:
            BGR color tuple
        """
        # Generate consistent color based on class ID
        np.random.seed(class_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
    
    def export_to_tflite(self, output_path: str, 
                        quantize: bool = True):
        """
        Export model to TensorFlow Lite for edge deployment
        
        Args:
            output_path: Path to save TFLite model
            quantize: Whether to apply INT8 quantization
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to TFLite
        if quantize:
            self.model.export(format='tflite', int8=True)
        else:
            self.model.export(format='tflite')
        
        print(f"Model exported to TFLite: {output_path}")
    
    def train(self, 
             data_yaml: str,
             epochs: int = 100,
             imgsz: int = 640,
             batch_size: int = 16,
             project: str = 'runs/train',
             name: str = 'zooplankton_detector',
             workers: int = 8,
             amp: bool = True):
        """
        Train YOLO26n model on custom dataset
        
        Args:
            data_yaml: Path to data configuration YAML
            epochs: Number of training epochs
            imgsz: Input image size
            batch_size: Training batch size
            project: Project directory
            name: Experiment name
            workers: Number of dataloader workers
            amp: Use Automatic Mixed Precision
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=project,
            name=name,
            device=self.device,
            workers=workers,
            patience=50,
            save=True,
            plots=True,
            amp=amp
        )
        
        return results
    
    def validate(self, data_yaml: str) -> Dict:
        """
        Validate model on validation dataset
        
        Args:
            data_yaml: Path to data configuration YAML
            
        Returns:
            Validation metrics
        """
        results = self.model.val(data=data_yaml)
        
        metrics = {
            'precision': results.box.p,
            'recall': results.box.r,
            'map50': results.box.map50,
            'map': results.box.map
        }
        
        return metrics
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_type': 'YOLO26n',
            'input_size': self.input_size,
            'device': self.device,
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'parameters': sum(p.numel() for p in self.model.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        }
        
        return info
