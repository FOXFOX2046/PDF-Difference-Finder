"""
Region generation from difference mask - detects and merges regions
"""
import numpy as np
import cv2
from typing import List, Tuple


def get_regions_from_mask(mask: np.ndarray, merge_distance: int = 50) -> List[Tuple[int, int, int, int]]:
    """
    Extract regions from binary mask using connected components with engineering filters
    
    ENGINEERING LOGIC:
    - Filter out small regions (< 0.2% of page area) - noise/anti-aliasing
    - Merge nearby regions to avoid fragmentation
    - Return all detected regions (no limit)
    
    Args:
        mask: Binary mask (H, W) with 255 for differences
        merge_distance: Maximum distance to merge nearby regions (pixels)
        
    Returns:
        List of bounding boxes as (x, y, width, height)
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Calculate page area for filtering
    page_area = mask.shape[0] * mask.shape[1]
    min_region_area = 0.002 * page_area  # 0.2% of page area
    
    regions = []
    
    # Skip background (label 0)
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = w * h
        
        # Filter out small regions (noise)
        if area > min_region_area:
            regions.append((x, y, w, h))
    
    # Merge nearby regions
    merged_regions = merge_nearby_regions(regions, merge_distance)
    
    # Sort by area (largest first) for consistency
    merged_regions = sorted(merged_regions, key=lambda r: r[2] * r[3], reverse=True)
    
    return merged_regions


def merge_nearby_regions(regions: List[Tuple[int, int, int, int]], 
                        max_distance: int) -> List[Tuple[int, int, int, int]]:
    """
    Merge regions that are close to each other
    
    Args:
        regions: List of (x, y, w, h) bounding boxes
        max_distance: Maximum distance to merge (pixels)
        
    Returns:
        Merged list of regions
    """
    if not regions:
        return []
    
    merged = []
    used = [False] * len(regions)
    
    for i, (x1, y1, w1, h1) in enumerate(regions):
        if used[i]:
            continue
        
        # Start with current region
        group = [(x1, y1, w1, h1)]
        used[i] = True
        
        # Find nearby regions
        changed = True
        while changed:
            changed = False
            for j, (x2, y2, w2, h2) in enumerate(regions):
                if used[j]:
                    continue
                
                # Check if this region is close to any in current group
                for gx, gy, gw, gh in group:
                    if boxes_nearby((x2, y2, w2, h2), (gx, gy, gw, gh), max_distance):
                        group.append((x2, y2, w2, h2))
                        used[j] = True
                        changed = True
                        break
        
        # Compute bounding box for merged group
        if group:
            min_x = min(x for x, _, _, _ in group)
            min_y = min(y for _, y, _, _ in group)
            max_x = max(x + w for x, _, w, _ in group)
            max_y = max(y + h for _, y, _, h in group)
            
            merged.append((min_x, min_y, max_x - min_x, max_y - min_y))
    
    return merged


def boxes_nearby(box1: Tuple[int, int, int, int], 
                 box2: Tuple[int, int, int, int], 
                 max_distance: int) -> bool:
    """Check if two bounding boxes are within max_distance"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Compute centers
    cx1 = x1 + w1 / 2
    cy1 = y1 + h1 / 2
    cx2 = x2 + w2 / 2
    cy2 = y2 + h2 / 2
    
    # Distance between centers
    dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    # Also check if boxes overlap or are very close
    overlap = not (x1 + w1 < x2 - max_distance or x2 + w2 < x1 - max_distance or
                   y1 + h1 < y2 - max_distance or y2 + h2 < y1 - max_distance)
    
    return dist <= max_distance or overlap


def get_fallback_region(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get fallback bounding box from mask if no regions detected
    
    Args:
        mask: Binary mask
        
    Returns:
        Bounding box (x, y, width, height) covering all white pixels
    """
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        # No differences, return empty box
        return (0, 0, 0, 0)
    
    min_y, max_y = np.min(coords[0]), np.max(coords[0])
    min_x, max_x = np.min(coords[1]), np.max(coords[1])
    
    return (min_x, min_y, max_x - min_x, max_y - min_y)

