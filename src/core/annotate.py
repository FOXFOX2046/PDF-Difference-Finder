"""
Annotation module - draws green overlay and red revision clouds
"""
import numpy as np
from typing import List, Tuple


def _require_cv2():
    try:
        import cv2
        return cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: cv2. Install 'opencv-python-headless' in your environment."
        ) from exc


def draw_revision_cloud_poly(
    img_bgr: np.ndarray,
    rect: Tuple[int, int, int, int],
    pad: int = 20,
    amp: int = 8,
    step: int = 14,
    thickness: int = 3,
    color: Tuple[int, int, int] = (0, 0, 255)  # BGR revision red
) -> np.ndarray:
    """
    Draw TRUE revision cloud using wavy continuous polyline
    
    Engineering standard revision cloud:
    - Continuous wavy line (not circles)
    - Surrounds the change region with padding
    - Professional appearance for drawing review
    
    Args:
        img_bgr: BGR image to draw on (OpenCV format)
        rect: Bounding box (x, y, w, h)
        pad: Padding around region (pixels)
        amp: Wave amplitude (pixels)
        step: Distance between wave points (pixels)
        thickness: Line thickness (pixels)
        color: BGR color tuple (default: red = 0,0,255)
        
    Returns:
        Image with revision cloud drawn (BGR format)
    """
    cv2 = _require_cv2()

    x, y, w, h = rect
    x -= pad
    y -= pad
    w += 2 * pad
    h += 2 * pad

    H, W = img_bgr.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, W - x)
    h = min(h, H - y)

    pts = []

    # Top edge
    for i, cx in enumerate(range(x, x + w + 1, step)):
        dy = -amp if i % 2 == 0 else amp
        pts.append((cx, y + dy))

    # Right edge
    for i, cy in enumerate(range(y, y + h + 1, step)):
        dx = amp if i % 2 == 0 else -amp
        pts.append((x + w + dx, cy))

    # Bottom edge
    for i, cx in enumerate(range(x + w, x - 1, -step)):
        dy = amp if i % 2 == 0 else -amp
        pts.append((cx, y + h + dy))

    # Left edge
    for i, cy in enumerate(range(y + h, y - 1, -step)):
        dx = -amp if i % 2 == 0 else amp
        pts.append((x + dx, cy))

    pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        img_bgr,
        [pts],
        isClosed=True,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA
    )
    return img_bgr


def draw_revision_cloud(img_bgr: np.ndarray, regions: List[Tuple[int, int, int, int]], 
                       cloud_thickness: int = 3) -> np.ndarray:
    """
    Draw revision clouds around all regions
    
    Cloud is drawn LAST (top-most layer).
    Uses wavy polyline for professional engineering appearance.
    
    Args:
        img_bgr: BGR image to draw on (OpenCV format)
        regions: List of bounding boxes (x, y, w, h)
        cloud_thickness: Thickness of cloud lines
        
    Returns:
        Annotated image (BGR format)
    """
    result = img_bgr.copy()
    
    if not regions:
        return result
    
    h, w = img_bgr.shape[:2]
    
    # Auto-scale parameters based on image size
    pad = max(20, int(min(w, h) * 0.015))  # ~1.5% padding
    amp = max(8, int(min(w, h) * 0.006))   # Wave amplitude
    step = max(14, int(min(w, h) * 0.01))  # Wave step
    
    for rect in regions:
        if rect[2] == 0 or rect[3] == 0:
            continue
        
        # Draw wavy revision cloud
        result = draw_revision_cloud_poly(
            result, 
            rect, 
            pad=pad, 
            amp=amp, 
            step=step,
            thickness=cloud_thickness,
            color=(0, 0, 255)  # BGR red
        )
    
    return result


def add_green_overlay(img_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """
    Add green semi-transparent overlay for differences
    
    Layer order: Base image → Green overlay
    
    Args:
        img_bgr: Base image (H, W, 3) BGR (OpenCV format)
        mask: Binary difference mask (H, W)
        alpha: Transparency of overlay (0.0-1.0)
        
    Returns:
        Image with green overlay (BGR format)
    """
    result = img_bgr.copy()
    
    # Create green overlay (BGR format: B=0, G=255, R=0)
    green_overlay = np.zeros_like(img_bgr)
    green_overlay[:, :, 1] = 255  # Green channel in BGR
    
    # Apply overlay only where mask is non-zero
    mask_3d = mask[:, :, np.newaxis] > 0
    overlay_mask = mask_3d.astype(float) * alpha
    
    result = (result * (1 - overlay_mask) + green_overlay * overlay_mask).astype(np.uint8)
    
    return result


def annotate_images_two_versions(img_a: np.ndarray, img_b: np.ndarray, 
                                 mask: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate annotated images with highlight only (NO raster clouds)
    
    REVISION CLOUDS ARE NOW PDF ANNOTATIONS ONLY (editable)
    This function only adds green highlight overlay
    
    STRICT DRAW ORDER:
    1. Base image (BGR from OpenCV)
    2. Green difference overlay (semi-transparent)
    
    IMPORTANT: Outputs are in BGR format (OpenCV). 
    Convert to RGB before displaying in Streamlit!
    
    Args:
        img_a: First image (BGR format from OpenCV)
        img_b: Second image (BGR format from OpenCV)
        mask: Binary difference mask
        regions: List of bounding boxes (not used for drawing, only for reference)
        
    Returns:
        Tuple of (img_a_highlight, img_a_highlight, img_b_highlight, img_b_highlight) in BGR format
        Note: Returns same image twice for backward compatibility
    """
    # VERSION: Highlight only (NO raster cloud - use PDF annotations instead)
    img_a_highlight = add_green_overlay(img_a, mask)
    img_b_highlight = add_green_overlay(img_b, mask)
    
    # Return same image for both versions (no cloud version on images)
    # Cloud version is now PDF annotations only
    return img_a_highlight, img_a_highlight, img_b_highlight, img_b_highlight


def annotate_images(img_a: np.ndarray, img_b: np.ndarray, 
                   mask: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Legacy wrapper - returns cloud version only for backward compatibility
    Use annotate_images_two_versions for full functionality
    """
    img_a_highlight, img_a_cloud, img_b_highlight, img_b_cloud = annotate_images_two_versions(
        img_a, img_b, mask, regions
    )
    return img_a_cloud, img_b_cloud
