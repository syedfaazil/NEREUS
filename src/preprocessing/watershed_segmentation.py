"""
Watershed Segmentation Module
Separates overlapping zooplankton specimens using watershed algorithm
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


class WatershedSegmenter:
    """
    Segments overlapping objects using watershed algorithm
    Particularly useful for separating touching/overlapping zooplankton
    """
    
    def __init__(self, 
                 min_distance: int = 20,
                 min_object_size: int = 100):
        """
        Initialize watershed segmenter
        
        Args:
            min_distance: Minimum distance between object centers
            min_object_size: Minimum size of objects to keep
        """
        self.min_distance = min_distance
        self.min_object_size = min_object_size
    
    def segment(self, image: np.ndarray, 
                return_masks: bool = False) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Segment overlapping objects using watershed
        
        Args:
            image: Input image (grayscale)
            return_masks: Whether to return individual object masks
            
        Returns:
            Segmented image and optionally list of individual masks
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Preprocess image
        binary = self._preprocess(image)
        
        # Apply watershed
        labels = self._apply_watershed(binary)
        
        # Create segmented image
        segmented = self._create_segmented_image(image, labels)
        
        # Extract individual masks if requested
        masks = None
        if return_masks:
            masks = self._extract_masks(labels)
        
        return segmented, masks
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for watershed segmentation
        
        Args:
            image: Input image
            
        Returns:
            Binary image
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Threshold using Otsu's method
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return binary
    
    def _apply_watershed(self, binary: np.ndarray) -> np.ndarray:
        """
        Apply watershed algorithm
        
        Args:
            binary: Binary image
            
        Returns:
            Labeled image with separated objects
        """
        # Compute distance transform
        distance = ndimage.distance_transform_edt(binary)
        
        # Find local maxima (object centers)
        local_max = peak_local_max(
            distance,
            min_distance=self.min_distance,
            labels=binary
        )
        
        # Create markers for watershed
        markers = np.zeros_like(binary, dtype=np.int32)
        for i, (y, x) in enumerate(local_max):
            markers[y, x] = i + 1
        
        # Apply watershed
        labels = watershed(-distance, markers, mask=binary)
        
        return labels
    
    def _create_segmented_image(self, image: np.ndarray, 
                                labels: np.ndarray) -> np.ndarray:
        """
        Create segmented image with boundaries
        
        Args:
            image: Original image
            labels: Labeled image from watershed
            
        Returns:
            Segmented image with boundaries drawn
        """
        # Create color image if grayscale
        if len(image.shape) == 2:
            segmented = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            segmented = image.copy()
        
        # Find boundaries
        boundaries = self._find_boundaries(labels)
        
        # Draw boundaries in red
        segmented[boundaries > 0] = [0, 0, 255]
        
        return segmented
    
    def _find_boundaries(self, labels: np.ndarray) -> np.ndarray:
        """
        Find boundaries between labeled regions
        
        Args:
            labels: Labeled image
            
        Returns:
            Binary image with boundaries
        """
        # Dilate each label and find overlaps
        boundaries = np.zeros_like(labels, dtype=np.uint8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        for label in np.unique(labels):
            if label == 0:  # Skip background
                continue
            
            mask = (labels == label).astype(np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            
            # Boundary is where dilated region overlaps with other labels
            boundary = dilated & (labels != label) & (labels != 0)
            boundaries |= boundary.astype(np.uint8)
        
        return boundaries
    
    def _extract_masks(self, labels: np.ndarray) -> List[np.ndarray]:
        """
        Extract individual object masks
        
        Args:
            labels: Labeled image
            
        Returns:
            List of binary masks for each object
        """
        masks = []
        
        for label in np.unique(labels):
            if label == 0:  # Skip background
                continue
            
            mask = (labels == label).astype(np.uint8) * 255
            
            # Check size
            if np.sum(mask > 0) >= self.min_object_size:
                masks.append(mask)
        
        return masks
    
    def extract_individual_objects(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract individual objects as separate images with bounding boxes
        
        Args:
            image: Input image
            
        Returns:
            List of (object_image, bounding_box) tuples
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Segment image
        _, masks = self.segment(image, return_masks=True)
        
        if masks is None:
            return []
        
        objects = []
        
        for mask in masks:
            # Find bounding box
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            # Get largest contour
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extract object
            object_img = image[y:y+h, x:x+w].copy()
            object_mask = mask[y:y+h, x:x+w]
            
            # Apply mask to object
            if len(object_img.shape) == 2:
                object_img = cv2.bitwise_and(object_img, object_img, mask=object_mask)
            else:
                object_img = cv2.bitwise_and(object_img, object_img, mask=object_mask)
            
            objects.append((object_img, (x, y, w, h)))
        
        return objects
    
    def count_objects(self, image: np.ndarray) -> int:
        """
        Count number of objects in image
        
        Args:
            image: Input image
            
        Returns:
            Number of objects detected
        """
        _, masks = self.segment(image, return_masks=True)
        
        if masks is None:
            return 0
        
        return len(masks)
    
    def get_object_statistics(self, image: np.ndarray) -> List[dict]:
        """
        Get statistics for each object
        
        Args:
            image: Input image
            
        Returns:
            List of dictionaries containing object statistics
        """
        objects = self.extract_individual_objects(image)
        
        statistics = []
        
        for obj_img, (x, y, w, h) in objects:
            stats = {
                'bbox': (x, y, w, h),
                'area': w * h,
                'width': w,
                'height': h,
                'aspect_ratio': w / h if h > 0 else 0,
                'mean_intensity': np.mean(obj_img[obj_img > 0]) if np.any(obj_img > 0) else 0,
                'std_intensity': np.std(obj_img[obj_img > 0]) if np.any(obj_img > 0) else 0
            }
            
            statistics.append(stats)
        
        return statistics
