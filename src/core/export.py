"""
Export module - saves marked pages to PDF or PNG
"""
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
from typing import List
import zipfile
import os


def images_to_pdf(images: List[np.ndarray], output_path: str, dpi: int = 150, compress: bool = True, quality: int = 85):
    """
    Convert list of images to PDF with optional compression
    
    Args:
        images: List of RGB images as numpy arrays (H, W, 3)
        output_path: Output PDF path
        dpi: Resolution for PDF pages (for metadata, actual resolution depends on image size)
        compress: Whether to use JPEG compression (True) or PNG (False)
        quality: JPEG quality (1-100, only used if compress=True)
    """
    if not images:
        return
    
    doc = fitz.open()
    
    for img_array in images:
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(img_array)
        
        # Convert to RGB if needed
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        # Save PIL image to bytes
        from io import BytesIO
        img_bytes = BytesIO()
        
        if compress:
            # Use JPEG compression for smaller file size
            pil_img.save(img_bytes, format='JPEG', quality=quality, optimize=True)
            img_format = 'jpeg'
        else:
            # Use PNG for lossless quality
            pil_img.save(img_bytes, format='PNG', optimize=True)
            img_format = 'png'
        
        img_bytes.seek(0)
        
        # Create PDF page
        width, height = pil_img.size
        rect = fitz.Rect(0, 0, width, height)
        page = doc.new_page(width=width, height=height)
        
        # Insert image with compression
        page.insert_image(rect, stream=img_bytes.read())
    
    # Save PDF with compression options
    doc.save(output_path, garbage=4, deflate=True, incremental=False)
    doc.close()


def save_images_as_pngs(images: List[np.ndarray], output_dir: str, prefix: str = "page"):
    """
    Save images as PNG files
    
    Args:
        images: List of RGB images as numpy arrays
        output_dir: Output directory
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_array in enumerate(images):
        pil_img = Image.fromarray(img_array)
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
        
        filename = f"{prefix}_{i+1:03d}.png"
        filepath = os.path.join(output_dir, filename)
        pil_img.save(filepath)


def create_zip_from_images(images: List[np.ndarray], output_path: str, prefix: str = "page"):
    """
    Create ZIP file containing PNG images
    
    Args:
        images: List of RGB images as numpy arrays
        output_path: Output ZIP file path
        prefix: Filename prefix for PNGs
    """
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, img_array in enumerate(images):
            pil_img = Image.fromarray(img_array)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            
            # Save to memory
            from io import BytesIO
            img_bytes = BytesIO()
            pil_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            filename = f"{prefix}_{i+1:03d}.png"
            zipf.writestr(filename, img_bytes.read())

