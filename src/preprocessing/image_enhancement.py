"""
Image Enhancement Module
Applies CLAHE, denoising, and other enhancement techniques
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class ImageEnhancer:
    """
    Enhances microscope images using various techniques:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Denoising
    - Sharpening
    - Brightness/Contrast adjustment
    """
    
    def __init__(self, 
                 clahe_clip_limit: float = 2.0,
                 clahe_tile_size: Tuple[int, int] = (8, 8),
                 denoise_strength: int = 10):
        """
        Initialize the enhancer
        
        Args:
            clahe_clip_limit: Clip limit for CLAHE
            clahe_tile_size: Tile grid size for CLAHE
            denoise_strength: Strength of denoising filter
        """
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.denoise_strength = denoise_strength
        
        # Create CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit,
            tileGridSize=clahe_tile_size
        )
    
    def enhance(self, image: np.ndarray, 
                apply_clahe: bool = True,
                apply_denoise: bool = True,
                apply_sharpen: bool = False) -> np.ndarray:
        """
        Apply enhancement pipeline to image
        
        Args:
            image: Input image (grayscale or RGB)
            apply_clahe: Whether to apply CLAHE
            apply_denoise: Whether to apply denoising
            apply_sharpen: Whether to apply sharpening
            
        Returns:
            Enhanced image
        """
        # Ensure image is in correct format
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        
        enhanced = image.copy()
        
        # Apply CLAHE for contrast enhancement
        if apply_clahe:
            enhanced = self._apply_clahe(enhanced)
        
        # Apply denoising
        if apply_denoise:
            enhanced = self._apply_denoise(enhanced)
        
        # Apply sharpening
        if apply_sharpen:
            enhanced = self._apply_sharpen(enhanced)
        
        return enhanced
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input image
            
        Returns:
            CLAHE enhanced image
        """
        if len(image.shape) == 2:
            # Grayscale image
            return self.clahe.apply(image)
        else:
            # Color image - convert to LAB and apply to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def _apply_denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising filter
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        if len(image.shape) == 2:
            # Grayscale - use fastNlMeansDenoising
            return cv2.fastNlMeansDenoising(
                image,
                None,
                h=self.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
        else:
            # Color - use fastNlMeansDenoisingColored
            return cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=self.denoise_strength,
                hColor=self.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
    
    def _apply_sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter
        
        Args:
            image: Input image
            
        Returns:
            Sharpened image
        """
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        return cv2.filter2D(image, -1, kernel)
    
    def adaptive_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive enhancement based on image statistics
        
        Args:
            image: Input image
            
        Returns:
            Adaptively enhanced image
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Calculate image statistics
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        # Determine enhancement strategy based on statistics
        enhanced = image.copy()
        
        # Low contrast image - apply stronger CLAHE
        if std_val < 30:
            clahe_strong = cv2.createCLAHE(
                clipLimit=3.0,
                tileGridSize=self.clahe_tile_size
            )
            if len(enhanced.shape) == 2:
                enhanced = clahe_strong.apply(enhanced)
            else:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe_strong.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Dark image - apply gamma correction
        elif mean_val < 100:
            gamma = 1.5
            enhanced = self._gamma_correction(enhanced, gamma)
        
        # Bright image - reduce brightness
        elif mean_val > 180:
            gamma = 0.7
            enhanced = self._gamma_correction(enhanced, gamma)
        
        # Apply standard CLAHE for normal images
        else:
            enhanced = self._apply_clahe(enhanced)
        
        # Always apply light denoising
        enhanced = self._apply_denoise(enhanced)
        
        return enhanced
    
    def _gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction
        
        Args:
            image: Input image
            gamma: Gamma value
            
        Returns:
            Gamma corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        
        return cv2.LUT(image, table)
    
    def normalize_illumination(self, image: np.ndarray, 
                               sigma: float = 50.0) -> np.ndarray:
        """
        Normalize uneven illumination using morphological operations
        
        Args:
            image: Input image
            sigma: Size of the structuring element
            
        Returns:
            Illumination normalized image
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Estimate background using morphological opening
        kernel_size = int(sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        if len(image.shape) == 2:
            background = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            # Subtract background
            normalized = cv2.subtract(image, background)
            # Rescale to full range
            normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        else:
            # Process each channel separately
            normalized = np.zeros_like(image)
            for i in range(image.shape[2]):
                background = cv2.morphologyEx(image[:, :, i], cv2.MORPH_OPEN, kernel)
                normalized[:, :, i] = cv2.subtract(image[:, :, i], background)
            normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def enhance_edges(self, image: np.ndarray, 
                     method: str = 'unsharp') -> np.ndarray:
        """
        Enhance edges in the image
        
        Args:
            image: Input image
            method: Enhancement method ('unsharp', 'laplacian')
            
        Returns:
            Edge enhanced image
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        if method == 'unsharp':
            # Unsharp masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
            enhanced = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        elif method == 'laplacian':
            # Laplacian edge enhancement
            if len(image.shape) == 2:
                laplacian = cv2.Laplacian(image, cv2.CV_64F)
                enhanced = image - laplacian.astype(np.uint8)
            else:
                enhanced = image.copy()
                for i in range(image.shape[2]):
                    laplacian = cv2.Laplacian(image[:, :, i], cv2.CV_64F)
                    enhanced[:, :, i] = image[:, :, i] - laplacian.astype(np.uint8)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
