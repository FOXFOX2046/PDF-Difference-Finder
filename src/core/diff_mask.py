"""
Difference mask generation - computes binary mask from two images
"""
import numpy as np
import cv2
from typing import Tuple


def compute_diff_mask(img_a: np.ndarray, img_b: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Compute binary difference mask from two BGR images
    
    Pipeline:
    1. absdiff(imgA, imgB)
    2. Convert to grayscale
    3. Threshold (mapped from sensitivity slider)
    4. Morphology (close / dilate) to merge fragments
    
    Args:
        img_a: First image (H, W, 3) BGR (OpenCV format)
        img_b: Second image (H, W, 3) BGR (OpenCV format)
        threshold: Sensitivity threshold (0.0-1.0), mapped to actual pixel difference
        
    Returns:
        Binary mask (H, W) with 255 for differences, 0 for no difference
    """
    # Ensure images have same dimensions
    if img_a.shape != img_b.shape:
        h_min = min(img_a.shape[0], img_b.shape[0])
        w_min = min(img_a.shape[1], img_b.shape[1])
        img_a = img_a[:h_min, :w_min]
        img_b = img_b[:h_min, :w_min]
    
    # Step 1: Absolute difference
    diff = cv2.absdiff(img_a, img_b)
    
    # Step 2: Convert to grayscale (works for BGR)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Threshold
    # Map sensitivity (0.0-1.0) to actual threshold value (0-255)
    # Lower sensitivity = higher threshold (less sensitive)
    # Higher sensitivity = lower threshold (more sensitive)
    thresh_value = int(255 * (1.0 - threshold))
    _, binary = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
    
    # Step 4: Morphology operations to merge fragments
    kernel_size = max(3, int(min(img_a.shape[:2]) * 0.005))  # Adaptive kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close: dilate then erode (merge nearby regions)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Dilate slightly to ensure regions are connected
    dilated = cv2.dilate(closed, kernel, iterations=1)
    
    return dilated


def has_differences(mask: np.ndarray) -> bool:
    """Check if mask contains any differences"""
    return np.any(mask > 0)

