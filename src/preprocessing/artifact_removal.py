"""
Artifact Removal Module
Removes bubbles, debris, and other artifacts from microscope images
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from scipy import ndimage


class ArtifactRemover:
    """
    Removes common artifacts from microscope images:
    - Bubbles
    - Debris
    - Dust particles
    - Edge artifacts
    """
    
    def __init__(self, 
                 min_object_size: int = 50,
                 max_object_size: Optional[int] = None,
                 circularity_threshold: float = 0.7):
        """
        Initialize artifact remover
        
        Args:
            min_object_size: Minimum size (pixels) to keep
            max_object_size: Maximum size (pixels) to keep
            circularity_threshold: Threshold for bubble detection (0-1)
        """
        self.min_object_size = min_object_size
        self.max_object_size = max_object_size
        self.circularity_threshold = circularity_threshold
    
    def remove_artifacts(self, image: np.ndarray, 
                        remove_bubbles: bool = True,
                        remove_debris: bool = True,
                        remove_edge_artifacts: bool = True) -> np.ndarray:
        """
        Remove artifacts from image
        
        Args:
            image: Input image
            remove_bubbles: Remove circular bubble artifacts
            remove_debris: Remove small debris
            remove_edge_artifacts: Remove edge artifacts
            
        Returns:
            Cleaned image
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        cleaned = image.copy()
        
        # Remove bubbles (highly circular artifacts)
        if remove_bubbles:
            cleaned = self._remove_bubbles(cleaned)
        
        # Remove small debris
        if remove_debris:
            cleaned = self._remove_debris(cleaned)
        
        # Remove edge artifacts
        if remove_edge_artifacts:
            cleaned = self._remove_edge_artifacts(cleaned)
        
        return cleaned
    
    def _remove_bubbles(self, image: np.ndarray) -> np.ndarray:
        """
        Remove bubble artifacts (highly circular objects)
        
        Args:
            image: Input image
            
        Returns:
            Image with bubbles removed
        """
        # Threshold image.
        # ZooScan images: dark specimens on a WHITE background.
        # THRESH_BINARY: background (bright) → 255, specimens (dark) → 0.
        # Bubbles are bright circular rings that also threshold to 255 —
        # so they appear in the binary and circularity check correctly catches them.
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for bubbles
        bubble_mask = np.zeros_like(image)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_object_size:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # If highly circular, mark as bubble
            if circularity > self.circularity_threshold:
                cv2.drawContours(bubble_mask, [contour], -1, 255, -1)
        
        # Inpaint bubbles
        if np.any(bubble_mask):
            result = cv2.inpaint(image, bubble_mask, 3, cv2.INPAINT_TELEA)
        else:
            result = image
        
        return result
    
    def _remove_debris(self, image: np.ndarray) -> np.ndarray:
        """
        Remove small debris particles
        
        Args:
            image: Input image
            
        Returns:
            Image with debris removed
        """
        # Apply morphological opening to remove small objects
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Find small disconnected regions
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Label connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Create mask for debris
        debris_mask = np.zeros_like(image)
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Mark small objects as debris
            if area < self.min_object_size:
                debris_mask[labels == i] = 255
        
        # Inpaint debris
        if np.any(debris_mask):
            result = cv2.inpaint(image, debris_mask, 3, cv2.INPAINT_TELEA)
        else:
            result = image
        
        return result
    
    def _remove_edge_artifacts(self, image: np.ndarray, 
                               border_size: int = 10) -> np.ndarray:
        """
        Remove artifacts at image edges
        
        Args:
            image: Input image
            border_size: Size of border to check
            
        Returns:
            Image with edge artifacts removed
        """
        # Create mask for edge region
        mask = np.zeros_like(image)
        h, w = image.shape[:2]
        
        # Mark border region
        mask[:border_size, :] = 255
        mask[-border_size:, :] = 255
        mask[:, :border_size] = 255
        mask[:, -border_size:] = 255
        
        # Find objects touching border
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Label connected components
        num_labels, labels = cv2.connectedComponents(binary)
        
        # Find labels that touch border
        border_labels = set()
        for i in range(1, num_labels):
            if np.any((labels == i) & (mask > 0)):
                border_labels.add(i)
        
        # Create mask for border objects
        border_mask = np.zeros_like(image)
        for label in border_labels:
            border_mask[labels == label] = 255
        
        # Inpaint border artifacts
        if np.any(border_mask):
            result = cv2.inpaint(image, border_mask, 3, cv2.INPAINT_TELEA)
        else:
            result = image
        
        return result
    
    def detect_and_remove_outliers(self, image: np.ndarray, 
                                   z_threshold: float = 3.0) -> np.ndarray:
        """
        Remove outlier pixels based on statistical analysis
        
        Args:
            image: Input image
            z_threshold: Z-score threshold for outlier detection
            
        Returns:
            Image with outliers removed
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Calculate statistics
        mean = np.mean(image)
        std = np.std(image)
        
        # Find outliers
        z_scores = np.abs((image - mean) / std)
        outlier_mask = (z_scores > z_threshold).astype(np.uint8) * 255
        
        # Inpaint outliers
        if np.any(outlier_mask):
            result = cv2.inpaint(image, outlier_mask, 3, cv2.INPAINT_TELEA)
        else:
            result = image
        
        return result
    
    def remove_background_gradient(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background gradient/illumination artifacts
        
        Args:
            image: Input image
            
        Returns:
            Image with background gradient removed
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Estimate background using large Gaussian blur
        background = cv2.GaussianBlur(image, (0, 0), sigmaX=50, sigmaY=50)
        
        # Subtract background
        result = cv2.subtract(image, background)
        
        # Normalize to full range
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        
        return result
