"""
Zooplankton Pipeline
Complete end-to-end pipeline for zooplankton detection and classification
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing import ImageStandardizer, ImageEnhancer, ArtifactRemover, WatershedSegmenter
from detection import YOLODetector
from classification import MobileNetClassifier


class ZooplanktonPipeline:
    """
    Complete Zooplankton pipeline integrating:
    1. Image preprocessing (standardization, enhancement, artifact removal)
    2. Object detection (YOLOv5n)
    3. Species classification (MobileNetV3)
    """
    
    def __init__(self,
                 detector_path: Optional[str] = None,
                 classifier_path: Optional[str] = None,
                 num_classes: int = 127,
                 device: str = 'cpu',
                 enable_preprocessing: bool = True,
                 enable_watershed: bool = True):
        """
        Initialize Zooplankton pipeline
        
        Args:
            detector_path: Path to trained YOLO detector weights
            classifier_path: Path to trained classifier weights
            num_classes: Number of species classes
            device: Device for inference ('cpu', 'cuda')
            enable_preprocessing: Whether to apply preprocessing
            enable_watershed: Whether to apply watershed segmentation
        """
        self.device = device
        self.enable_preprocessing = enable_preprocessing
        self.enable_watershed = enable_watershed
        
        # Initialize preprocessing modules
        if enable_preprocessing:
            self.standardizer = ImageStandardizer(normalize=False)
            self.enhancer = ImageEnhancer()
            self.artifact_remover = ArtifactRemover()
        
        if enable_watershed:
            self.watershed = WatershedSegmenter()
        
        # Initialize detection module
        self.detector = YOLODetector(
            model_path=detector_path,
            device=device
        )
        
        # Initialize classification module
        self.classifier = MobileNetClassifier(
            model_path=classifier_path,
            num_classes=num_classes,
            device=device
        )
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_detections': 0,
            'species_counts': defaultdict(int),
            'processing_times': []
        }
    
    def process_image(self, 
                     image: np.ndarray,
                     return_visualization: bool = False,
                     top_k_classes: int = 3) -> Dict:
        """
        Process a single image through the complete pipeline
        
        Args:
            image: Input image
            return_visualization: Whether to return visualization
            top_k_classes: Number of top class predictions per detection
            
        Returns:
            Dictionary containing results
        """
        start_time = time.time()
        
        # Step 1: Preprocessing
        if self.enable_preprocessing:
            preprocessed = self._preprocess_image(image)
        else:
            preprocessed = image.copy()
        
        # Step 2: Detection
        detections, crops = self.detector.detect(preprocessed, return_crops=True)
        
        # Step 3: Classification
        classifications = []
        if crops:
            for crop in crops:
                predictions = self.classifier.classify(crop, top_k=top_k_classes)
                classifications.append(predictions)
        
        # Combine results
        results = []
        for i, det in enumerate(detections):
            result = {
                'bbox': det['bbox'],
                'confidence': det['confidence'],
                'detection_class': det['class'],
                'species_predictions': classifications[i] if i < len(classifications) else [],
                'top_species': classifications[i][0] if i < len(classifications) and classifications[i] else None
            }
            results.append(result)
        
        # Update statistics
        processing_time = time.time() - start_time
        self.stats['total_processed'] += 1
        self.stats['total_detections'] += len(detections)
        self.stats['processing_times'].append(processing_time)
        
        # Update species counts
        for result in results:
            if result['top_species']:
                species_id = result['top_species'][0]
                self.stats['species_counts'][species_id] += 1
        
        # Prepare output
        output = {
            'num_detections': len(detections),
            'detections': results,
            'processing_time': processing_time,
            'preprocessed_image': preprocessed
        }
        
        # Add visualization if requested
        if return_visualization:
            vis_image = self._visualize_results(image, results)
            output['visualization'] = vis_image
        
        return output
    
    def process_batch(self,
                     images: List[np.ndarray],
                     batch_size: int = 8,
                     return_visualizations: bool = False) -> List[Dict]:
        """
        Process batch of images
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            return_visualizations: Whether to return visualizations
            
        Returns:
            List of result dictionaries
        """
        all_results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            for image in batch:
                result = self.process_image(
                    image,
                    return_visualization=return_visualizations
                )
                all_results.append(result)
        
        return all_results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing pipeline
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Apply enhancement
        enhanced = self.enhancer.adaptive_enhancement(image)
        
        # Remove artifacts
        cleaned = self.artifact_remover.remove_artifacts(enhanced)
        
        return cleaned
    
    def _visualize_results(self, 
                          image: np.ndarray,
                          results: List[Dict]) -> np.ndarray:
        """
        Visualize detection and classification results
        
        Args:
            image: Original image
            results: List of result dictionaries
            
        Returns:
            Visualization image
        """
        vis_image = image.copy()
        
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            if result['top_species']:
                species_id, conf = result['top_species']
                species_name = self.classifier.get_class_name(species_id)
                label = f"{species_name}: {conf:.2f}"
            else:
                label = f"Det: {result['confidence']:.2f}"
            
            # Draw label
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(vis_image, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), (0, 255, 0), -1)
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return vis_image
    
    def count_species(self, image: np.ndarray) -> Dict[str, int]:
        """
        Count species in image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary mapping species names to counts
        """
        result = self.process_image(image)
        
        species_counts = defaultdict(int)
        for det in result['detections']:
            if det['top_species']:
                species_id = det['top_species'][0]
                species_name = self.classifier.get_class_name(species_id)
                species_counts[species_name] += 1
        
        return dict(species_counts)
    
    def calculate_diversity_indices(self, species_counts: Dict[str, int]) -> Dict:
        """
        Calculate biodiversity indices
        
        Args:
            species_counts: Dictionary of species counts
            
        Returns:
            Dictionary of diversity indices
        """
        total = sum(species_counts.values())
        
        if total == 0:
            return {
                'species_richness': 0,
                'shannon_index': 0.0,
                'simpson_index': 0.0,
                'evenness': 0.0
            }
        
        # Species richness
        richness = len(species_counts)
        
        # Shannon diversity index
        shannon = 0.0
        for count in species_counts.values():
            if count > 0:
                p = count / total
                shannon -= p * np.log(p)
        
        # Simpson diversity index
        simpson = 0.0
        for count in species_counts.values():
            if count > 0:
                p = count / total
                simpson += p ** 2
        simpson = 1 - simpson
        
        # Evenness
        evenness = shannon / np.log(richness) if richness > 1 else 0.0
        
        return {
            'species_richness': richness,
            'shannon_index': float(shannon),
            'simpson_index': float(simpson),
            'evenness': float(evenness),
            'total_individuals': total
        }
    
    def get_statistics(self) -> Dict:
        """
        Get pipeline statistics
        
        Returns:
            Dictionary of statistics
        """
        avg_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
        
        return {
            'total_images_processed': self.stats['total_processed'],
            'total_detections': self.stats['total_detections'],
            'average_processing_time': avg_time,
            'fps': 1.0 / avg_time if avg_time > 0 else 0,
            'species_distribution': dict(self.stats['species_counts'])
        }
    
    def reset_statistics(self):
        """Reset pipeline statistics"""
        self.stats = {
            'total_processed': 0,
            'total_detections': 0,
            'species_counts': defaultdict(int),
            'processing_times': []
        }
    
    def export_results(self, 
                      results: List[Dict],
                      output_path: str,
                      format: str = 'csv'):
        """
        Export results to file
        
        Args:
            results: List of result dictionaries
            output_path: Path to output file
            format: Output format ('csv', 'json')
        """
        import json
        import csv
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'image_id', 'detection_id', 'bbox_x1', 'bbox_y1', 
                    'bbox_x2', 'bbox_y2', 'detection_confidence',
                    'species_id', 'species_name', 'species_confidence'
                ])
                
                for img_id, result in enumerate(results):
                    for det_id, det in enumerate(result['detections']):
                        x1, y1, x2, y2 = det['bbox']
                        
                        if det['top_species']:
                            species_id, species_conf = det['top_species']
                            species_name = self.classifier.get_class_name(species_id)
                        else:
                            species_id = -1
                            species_name = 'Unknown'
                            species_conf = 0.0
                        
                        writer.writerow([
                            img_id, det_id, x1, y1, x2, y2,
                            det['confidence'], species_id, species_name,
                            species_conf
                        ])
        
        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        print(f"Results exported to {output_path}")
    
    def get_pipeline_info(self) -> Dict:
        """
        Get pipeline configuration information
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'device': self.device,
            'preprocessing_enabled': self.enable_preprocessing,
            'watershed_enabled': self.enable_watershed,
            'detector_info': self.detector.get_model_info(),
            'classifier_info': self.classifier.get_model_info()
        }
