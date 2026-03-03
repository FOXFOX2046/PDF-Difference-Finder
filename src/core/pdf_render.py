"""
PDF rendering module - converts PDF pages to RGB images
"""
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


def pdf_to_images(pdf_path: str, page_num: Optional[int] = None, dpi: int = 200) -> List[np.ndarray]:
    """
    Convert PDF pages to BGR images (OpenCV format)
    
    Args:
        pdf_path: Path to PDF file
        page_num: Specific page number (0-indexed), None for all pages
        dpi: Resolution for rendering (default 200 for high quality diff detection)
        
    Returns:
        List of BGR images as numpy arrays (H, W, 3) - OpenCV format
    """
    import cv2
    
    doc = fitz.open(pdf_path)
    images = []
    
    if page_num is not None:
        pages = [page_num]
    else:
        pages = range(len(doc))
    
    for page_idx in pages:
        if page_idx >= len(doc):
            continue
            
        page = doc[page_idx]
        # Render page to pixmap (RGB)
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # Scale factor
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to numpy array (RGB from PyMuPDF)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        
        # Ensure RGB (3 channels)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        elif img_array.shape[2] == 1:
            img_array = np.repeat(img_array, 3, axis=2)
        
        # Convert RGB to BGR for OpenCV processing
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        images.append(img_bgr)
    
    doc.close()
    return images


def get_pdf_page_count(pdf_path: str) -> int:
    """Get total number of pages in PDF"""
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count

