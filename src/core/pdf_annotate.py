"""
PDF annotation module - adds editable revision cloud annotations to PDF
"""
import fitz  # PyMuPDF
import numpy as np
import math
from typing import List, Tuple, Dict
from pathlib import Path


def px_box_to_pdf_rect(
    x_px: int, y_px: int, w_px: int, h_px: int,
    img_w: int, img_h: int,
    page: fitz.Page,
    pad_px: int = 20
) -> fitz.Rect:
    """
    Convert a region box in rendered image pixel coordinates to a PDF Rect (points).
    Assumes the image was rendered from THIS SAME page using PyMuPDF get_pixmap().
    IMPORTANT: No Y flip.
    
    Args:
        x_px, y_px, w_px, h_px: Region box in pixels
        img_w, img_h: Rendered image dimensions (MUST match diff detection render)
        page: PyMuPDF Page object
        pad_px: Padding in pixels
        
    Returns:
        fitz.Rect in PDF points
    """
    pdf_w = page.rect.width   # points
    pdf_h = page.rect.height  # points

    # Prevent division by zero
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image dimensions: img_w={img_w}, img_h={img_h}")
    
    sx = pdf_w / float(img_w)
    sy = pdf_h / float(img_h)

    # padding in pixel space
    x0_px = x_px - pad_px
    y0_px = y_px - pad_px
    x1_px = x_px + w_px + pad_px
    y1_px = y_px + h_px + pad_px

    # clamp in pixel space
    x0_px = max(0, x0_px)
    y0_px = max(0, y0_px)
    x1_px = min(img_w, x1_px)
    y1_px = min(img_h, y1_px)

    # scale to PDF points (NO Y FLIP)
    x0 = x0_px * sx
    y0 = y0_px * sy
    x1 = x1_px * sx
    y1 = y1_px * sy

    return fitz.Rect(x0, y0, x1, y1)


def add_revision_cloud_annotations(
    pdf_path: str,
    output_path: str,
    page_regions: Dict[int, List[Tuple[int, int, int, int]]],
    page_dimensions: Dict[int, Dict],
    dpi: int = 200
):
    """
    Add editable revision cloud annotations to PDF
    
    Args:
        pdf_path: Input PDF path
        output_path: Output PDF path
        page_regions: Dict mapping page_num -> list of (x,y,w,h) pixel boxes
        page_dimensions: Dict mapping page_num -> {img_w, img_h, pdf_w, pdf_h}
        dpi: DPI used for rendering (for reference)
    """
    doc = fitz.open(pdf_path)
    
    total_annotations = 0
    
    for page_num, regions in page_regions.items():
        if page_num >= len(doc):
            continue
        
        if not regions:  # Skip empty region lists
            continue
        
        page = doc[page_num]
        dims = page_dimensions[page_num]
        
        # Use EXACT rendered image size used for diff detection
        img_w = dims['img_width']
        img_h = dims['img_height']
        
        # Debug: print dimensions once per page
        print(f"Page {page_num + 1}: IMG_W_H(px): {img_w}, {img_h}")
        print(f"Page {page_num + 1}: PDF_W_H(pt): {page.rect.width}, {page.rect.height}")
        
        page_annot_count = 0
        
        for pixel_box in regions:
            x, y, w, h = pixel_box
            
            # BOX-RELATIVE padding: 8% of larger dimension, clamped to [8, 40]
            pad_px = int(0.08 * max(w, h))
            pad_px = max(8, min(40, pad_px))
            
            # Debug: print region
            print(f"Page {page_num + 1}: REGION_PX: x={x}, y={y}, w={w}, h={h}, pad={pad_px}")
            
            # Convert pixel box to PDF Rect (NO Y FLIP)
            try:
                rect = px_box_to_pdf_rect(x, y, w, h, img_w, img_h, page, pad_px=pad_px)
            except Exception as e:
                print(f"Page {page_num + 1}: ERROR - Failed to convert pixel box to PDF rect: {e}")
                continue
            
            # Validate rect coordinates
            try:
                # Check if rect values are valid numbers
                if (math.isnan(rect.x0) or math.isnan(rect.y0) or math.isnan(rect.x1) or math.isnan(rect.y1) or
                    math.isinf(rect.x0) or math.isinf(rect.y0) or math.isinf(rect.x1) or math.isinf(rect.y1) or
                    rect.x0 >= rect.x1 or rect.y0 >= rect.y1 or
                    rect.x0 < 0 or rect.y0 < 0 or rect.x1 > page.rect.width or rect.y1 > page.rect.height):
                    print(f"Page {page_num + 1}: WARNING - Invalid rect coordinates, skipping: x0={rect.x0}, y0={rect.y0}, x1={rect.x1}, y1={rect.y1}")
                    continue
                
                # Debug: print PDF rect
                print(f"Page {page_num + 1}: RECT_PDF: x0={rect.x0}, y0={rect.y0}, x1={rect.x1}, y1={rect.y1}")
            except Exception as e:
                print(f"Page {page_num + 1}: ERROR - Failed to validate/print rect: {e}")
                continue
            
            # Add rectangle annotation
            try:
                annot = page.add_rect_annot(rect)
            except Exception as e:
                try:
                    rect_str = f"x0={rect.x0}, y0={rect.y0}, x1={rect.x1}, y1={rect.y1}"
                except:
                    rect_str = "rect (unable to read coordinates)"
                print(f"Page {page_num + 1}: ERROR - Failed to add annotation: {e}, {rect_str}")
                continue
            
            # Set red color (RGB) - PDF uses 0-1 range
            annot.set_colors(stroke=(1, 0, 0))
            
            # Set border width
            annot.set_border(width=2)
            
            # Set opacity
            annot.set_opacity(1.0)
            
            # Set info metadata (includes content for comments panel)
            try:
                annot.set_info({"title": "PDFDIFF", "subject": "Revision Cloud", "content": "PDFDIFF_CLOUD"})
            except Exception:
                # Fallback: set info fields individually
                try:
                    annot.set_info({"title": "PDFDIFF", "subject": "Revision Cloud"})
                    annot.info["content"] = "PDFDIFF_CLOUD"
                except Exception:
                    pass
            
            # MUST CALL update() to commit the annotation
            annot.update()
            
            # Try to set CLOUDY border effect using low-level PDF operations
            try:
                xref = annot.xref
                if xref > 0:
                    # Get current annotation dictionary
                    annot_dict = doc.xref_object(xref)
                    
                    # Add Border Effect entry for cloudy appearance
                    # /BE << /S /C /I 2 >>
                    # S = Style (C = Cloudy), I = Intensity
                    if "/BE" not in annot_dict:
                        new_dict = annot_dict.replace(
                            "/Subtype/Square",
                            "/Subtype/Square/BE<</S/C/I 2>>"
                        )
                        doc.update_object(xref, new_dict)
            except Exception:
                # If cloudy effect fails, annotation is still valid with solid border
                pass
            
            page_annot_count += 1
            total_annotations += 1
        
        if page_annot_count > 0:
            print(f"Page {page_num + 1}: Added {page_annot_count} annotation(s)")
        else:
            print(f"Page {page_num + 1}: WARNING - No annotations added")
    
    # Save annotated PDF
    try:
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        doc.save(output_path, garbage=4, deflate=True, incremental=False)
    except Exception as e:
        raise OSError(f"Failed to save PDF to {output_path}: {e}")
    finally:
        doc.close()
    
    print(f"Total: Added {total_annotations} annotations to {output_path}")
    return total_annotations


def create_annotated_pdfs(
    pdf_a_path: str,
    pdf_b_path: str,
    output_a_path: str,
    output_b_path: str,
    all_regions_a: List[List[Tuple[int, int, int, int]]],
    all_regions_b: List[List[Tuple[int, int, int, int]]],
    all_dimensions_a: List[Dict],
    all_dimensions_b: List[Dict],
    dpi: int = 200
):
    """
    Create annotated PDFs for both A and B with revision clouds
    
    Args:
        pdf_a_path: Input PDF A path
        pdf_b_path: Input PDF B path
        output_a_path: Output PDF A path
        output_b_path: Output PDF B path
        all_regions_a: List of regions per page for PDF A
        all_regions_b: List of regions per page for PDF B
        all_dimensions_a: List of dimensions per page for PDF A
        all_dimensions_b: List of dimensions per page for PDF B
        dpi: DPI used for rendering
    """
    # Convert lists to dicts
    page_regions_a = {i: regions for i, regions in enumerate(all_regions_a) if regions}
    page_regions_b = {i: regions for i, regions in enumerate(all_regions_b) if regions}
    
    page_dimensions_a = {i: dims for i, dims in enumerate(all_dimensions_a)}
    page_dimensions_b = {i: dims for i, dims in enumerate(all_dimensions_b)}
    
    # Add annotations to PDF A
    if page_regions_a:
        add_revision_cloud_annotations(
            pdf_a_path, output_a_path, page_regions_a, page_dimensions_a, dpi
        )
    else:
        # No annotations, just copy
        import shutil
        shutil.copy2(pdf_a_path, output_a_path)
    
    # Add annotations to PDF B
    if page_regions_b:
        add_revision_cloud_annotations(
            pdf_b_path, output_b_path, page_regions_b, page_dimensions_b, dpi
        )
    else:
        # No annotations, just copy
        import shutil
        shutil.copy2(pdf_b_path, output_b_path)


def get_page_dimensions(pdf_path: str, page_num: int, dpi: int = 200) -> Dict:
    """
    Get both pixel and PDF dimensions for a page
    
    Args:
        pdf_path: PDF file path
        page_num: Page number (0-indexed)
        dpi: DPI for rendering
        
    Returns:
        Dict with img_width, img_height, pdf_width, pdf_height
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # PDF dimensions in points
    rect = page.rect
    pdf_width = rect.width
    pdf_height = rect.height
    
    # Image dimensions at given DPI
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_width = pix.width
    img_height = pix.height
    
    doc.close()
    
    return {
        'img_width': img_width,
        'img_height': img_height,
        'pdf_width': pdf_width,
        'pdf_height': pdf_height
    }
