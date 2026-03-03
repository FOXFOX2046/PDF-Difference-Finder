"""
Main Streamlit application for PDF difference comparison
Supports single pair and batch processing modes
"""
import streamlit as st
import numpy as np
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict

from src.core.pdf_render import pdf_to_images, get_pdf_page_count
from src.core.diff_mask import compute_diff_mask, has_differences
from src.core.regions import get_regions_from_mask, get_fallback_region
from src.core.annotate import annotate_images_two_versions
from src.core.export import images_to_pdf, create_zip_from_images
from src.core.pdf_annotate import create_annotated_pdfs, get_page_dimensions


def process_pdf_pair(
    pdf_a_path: str,
    pdf_b_path: str,
    base_name_a: str,
    base_name_b: str,
    sensitivity: float,
    dpi: int = 200
) -> Dict:
    """
    Process a single PDF pair and return results
    
    Returns:
        Dict with keys: highlight_a, highlight_b, regions_a, regions_b, 
                       dimensions_a, dimensions_b, tmp_paths
    """
    # Get page counts
    pages_a = get_pdf_page_count(pdf_a_path)
    pages_b = get_pdf_page_count(pdf_b_path)
    
    if pages_a == 0 or pages_b == 0:
        return None
    
    # Process all pages
    all_pages_a = pdf_to_images(pdf_a_path, dpi=dpi)
    all_pages_b = pdf_to_images(pdf_b_path, dpi=dpi)
    
    all_highlight_a = []
    all_highlight_b = []
    all_regions_a = []
    all_regions_b = []
    all_dimensions_a = []
    all_dimensions_b = []
    
    for i in range(min(len(all_pages_a), len(all_pages_b))):
        page_img_a = all_pages_a[i]
        page_img_b = all_pages_b[i]
        page_mask = compute_diff_mask(page_img_a, page_img_b, threshold=sensitivity)
        page_regions = get_regions_from_mask(page_mask)
        
        if not page_regions and has_differences(page_mask):
            fallback = get_fallback_region(page_mask)
            if fallback[2] > 0 and fallback[3] > 0:
                page_regions = [fallback]
        
        h_a, c_a, h_b, c_b = annotate_images_two_versions(
            page_img_a, page_img_b, page_mask, page_regions
        )
        all_highlight_a.append(h_a)
        all_highlight_b.append(h_b)
        all_regions_a.append(page_regions)
        all_regions_b.append(page_regions)
        
        dims_a = get_page_dimensions(pdf_a_path, i, dpi=dpi)
        dims_b = get_page_dimensions(pdf_b_path, i, dpi=dpi)
        all_dimensions_a.append(dims_a)
        all_dimensions_b.append(dims_b)
    
    return {
        'highlight_a': all_highlight_a,
        'highlight_b': all_highlight_b,
        'regions_a': all_regions_a,
        'regions_b': all_regions_b,
        'dimensions_a': all_dimensions_a,
        'dimensions_b': all_dimensions_b,
        'base_name_a': base_name_a,
        'base_name_b': base_name_b,
        'pdf_a_path': pdf_a_path,
        'pdf_b_path': pdf_b_path
    }


def create_output_files(result: Dict, output_dir: str, is_batch: bool = False, compress: bool = True, quality: int = 85) -> List[str]:
    """
    Create output PDF files for a processed pair
    
    Args:
        result: Processing result dictionary
        output_dir: Output directory
        is_batch: Whether this is batch processing (use Original/Revised labels)
        compress: Whether to use JPEG compression (True) or PNG (False)
        quality: JPEG quality (1-100, only used if compress=True)
    
    Returns:
        List of output file paths
    """
    import cv2
    
    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    
    base_name_a = result['base_name_a']
    base_name_b = result['base_name_b']
    
    # Use Original/Revised labels for batch mode, A/B for single pair mode
    if is_batch:
        label_a = "Original"
        label_b = "Revised"
    else:
        label_a = "A"
        label_b = "B"
    
    # Create highlight PDFs
    highlight_a_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in result['highlight_a']]
    highlight_b_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in result['highlight_b']]
    
    highlight_a_path = os.path.join(output_dir, f"{base_name_a}_{label_a}_highlight.pdf")
    highlight_b_path = os.path.join(output_dir, f"{base_name_b}_{label_b}_highlight.pdf")
    
    images_to_pdf(highlight_a_rgb, highlight_a_path, compress=compress, quality=quality)
    images_to_pdf(highlight_b_rgb, highlight_b_path, compress=compress, quality=quality)
    output_files.extend([highlight_a_path, highlight_b_path])
    
    # Create annotated PDFs
    tmp_highlight_a = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_highlight_a_path = tmp_highlight_a.name
    tmp_highlight_a.close()
    
    tmp_highlight_b = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_highlight_b_path = tmp_highlight_b.name
    tmp_highlight_b.close()
    
    images_to_pdf(highlight_a_rgb, tmp_highlight_a_path, compress=compress, quality=quality)
    images_to_pdf(highlight_b_rgb, tmp_highlight_b_path, compress=compress, quality=quality)
    
    annot_a_path = os.path.join(output_dir, f"{base_name_a}_{label_a}_highlight_cloud.pdf")
    annot_b_path = os.path.join(output_dir, f"{base_name_b}_{label_b}_highlight_cloud.pdf")
    
    create_annotated_pdfs(
        tmp_highlight_a_path,
        tmp_highlight_b_path,
        annot_a_path,
        annot_b_path,
        result['regions_a'],
        result['regions_b'],
        result['dimensions_a'],
        result['dimensions_b'],
        dpi=200
    )
    
    output_files.extend([annot_a_path, annot_b_path])
    
    # Cleanup temp files
    try:
        os.unlink(tmp_highlight_a_path)
        os.unlink(tmp_highlight_b_path)
    except:
        pass
    
    return output_files


st.set_page_config(page_title="PDF Difference Finder", layout="wide")

st.title("📄 PDF Difference Finder")
st.markdown("Compare PDF files and automatically highlight differences with revision clouds")

# Sidebar app icon
_logo_path = Path(__file__).resolve().parent / "MadFoxLogo.png"
if _logo_path.exists():
    st.sidebar.image(str(_logo_path), use_container_width=True)

# Sidebar controls
st.sidebar.header("Controls")

# Processing mode selection
processing_mode = st.sidebar.radio(
    "Processing Mode",
    ["Single Pair", "Batch Mode"],
    help="Single Pair: Compare one pair of PDFs. Batch Mode: Process multiple PDF pairs at once."
)

# Sensitivity slider
sensitivity = st.sidebar.slider(
    "Sensitivity / Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="Higher values = more sensitive to differences"
)

# Compression settings
st.sidebar.markdown("---")
st.sidebar.markdown("**File Size Options**")
use_compression = st.sidebar.checkbox(
    "Use JPEG Compression",
    value=True,
    help="Enable JPEG compression to reduce file size (slight quality loss)"
)
if use_compression:
    jpeg_quality = st.sidebar.slider(
        "JPEG Quality",
        min_value=60,
        max_value=100,
        value=85,
        step=5,
        help="Higher quality = larger file size (85 recommended)"
    )
else:
    jpeg_quality = 85  # Default, not used when compression is off

st.sidebar.info("Visual mode: ON (default)")

# Main processing based on mode
if processing_mode == "Single Pair":
    # Single pair mode (original functionality)
    pdf_a = st.sidebar.file_uploader("PDF A (Original)", type=["pdf"], key="pdf_a")
    pdf_b = st.sidebar.file_uploader("PDF B (Compare)", type=["pdf"], key="pdf_b")
    
    if pdf_a is not None and pdf_b is not None:
        base_name_a = Path(pdf_a.name).stem
        base_name_b = Path(pdf_b.name).stem
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_a:
            tmp_a.write(pdf_a.read())
            tmp_a_path = tmp_a.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_b:
            tmp_b.write(pdf_b.read())
            tmp_b_path = tmp_b.name
        
        try:
            pages_a = get_pdf_page_count(tmp_a_path)
            pages_b = get_pdf_page_count(tmp_b_path)
            max_pages = max(pages_a, pages_b)
            
            if max_pages == 0:
                st.error("One or both PDFs have no pages")
            else:
                # Page selector
                selected_page = st.sidebar.selectbox(
                    "Select Page",
                    options=list(range(max_pages)),
                    format_func=lambda x: f"Page {x + 1}",
                    key="page_selector"
                )
                
                # Process selected page
                process_page = True
                if 'last_page' in st.session_state and st.session_state['last_page'] == selected_page and 'processed' in st.session_state:
                    if 'last_sensitivity' in st.session_state and st.session_state['last_sensitivity'] == sensitivity:
                        process_page = False
                
                if process_page:
                    st.session_state['last_page'] = selected_page
                    st.session_state['last_sensitivity'] = sensitivity
                    
                    with st.spinner("Processing PDFs..."):
                        try:
                            images_a = pdf_to_images(tmp_a_path, page_num=selected_page)
                            images_b = pdf_to_images(tmp_b_path, page_num=selected_page)
                        except Exception as e:
                            st.error(f"Error loading PDF pages: {e}")
                            images_a = []
                            images_b = []
                        
                        if images_a and images_b:
                            img_a = images_a[0]
                            img_b = images_b[0]
                            
                            mask = compute_diff_mask(img_a, img_b, threshold=sensitivity)
                            regions = get_regions_from_mask(mask)
                            
                            if not regions and has_differences(mask):
                                fallback_region = get_fallback_region(mask)
                                if fallback_region[2] > 0 and fallback_region[3] > 0:
                                    regions = [fallback_region]
                            
                            img_a_highlight, img_a_cloud, img_b_highlight, img_b_cloud = annotate_images_two_versions(
                                img_a, img_b, mask, regions
                            )
                            
                            st.session_state['img_a_highlight'] = img_a_highlight
                            st.session_state['img_b_highlight'] = img_b_highlight
                            st.session_state['regions'] = regions
                            st.session_state['has_diff'] = has_differences(mask)
                            st.session_state['processed'] = True
                            
                            # Process all pages for export
                            result = process_pdf_pair(tmp_a_path, tmp_b_path, base_name_a, base_name_b, sensitivity)
                            if result:
                                st.session_state['result'] = result
                        else:
                            st.error("Failed to load images from PDFs")
                
                # Display results
                if 'processed' in st.session_state and st.session_state['processed']:
                    st.info("💡 Revision clouds are PDF annotations (editable) - download annotated PDFs below")
                    
                    col1, col2 = st.columns(2)
                    import cv2
                    
                    with col1:
                        st.subheader("PDF A (Original) - Green Highlight")
                        if 'img_a_highlight' in st.session_state:
                            img_rgb = cv2.cvtColor(st.session_state['img_a_highlight'], cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, use_container_width=True)
                        
                        if st.session_state['has_diff']:
                            st.success(f"✓ {len(st.session_state['regions'])} region(s) detected")
                        else:
                            st.info("No differences detected")
                    
                    with col2:
                        st.subheader("PDF B (Compare) - Green Highlight")
                        if 'img_b_highlight' in st.session_state:
                            img_rgb = cv2.cvtColor(st.session_state['img_b_highlight'], cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, use_container_width=True)
                        
                        if st.session_state['has_diff']:
                            st.success(f"✓ {len(st.session_state['regions'])} region(s) detected")
                        else:
                            st.info("No differences detected")
                    
                    # Download section
                    st.sidebar.header("Download")
                    
                    if 'result' in st.session_state:
                        result = st.session_state['result']
                        import cv2
                        
                        # Highlight PDFs
                        st.sidebar.markdown("**📄 Highlight PDFs**")
                        highlight_a_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in result['highlight_a']]
                        highlight_b_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in result['highlight_b']]
                        
                        tmp_highlight_a = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp_highlight_a_path = tmp_highlight_a.name
                        tmp_highlight_a.close()
                        
                        tmp_highlight_b = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp_highlight_b_path = tmp_highlight_b.name
                        tmp_highlight_b.close()
                        
                        images_to_pdf(highlight_a_rgb, tmp_highlight_a_path, compress=use_compression, quality=jpeg_quality)
                        images_to_pdf(highlight_b_rgb, tmp_highlight_b_path, compress=use_compression, quality=jpeg_quality)
                        
                        with open(tmp_highlight_a_path, 'rb') as f:
                            pdf_a_highlight_bytes = f.read()
                        with open(tmp_highlight_b_path, 'rb') as f:
                            pdf_b_highlight_bytes = f.read()
                        
                        st.sidebar.download_button(
                            f"📄 {base_name_a}_A_highlight.pdf",
                            pdf_a_highlight_bytes,
                            file_name=f"{base_name_a}_A_highlight.pdf",
                            mime="application/pdf",
                            key="dl_a_highlight"
                        )
                        
                        st.sidebar.download_button(
                            f"📄 {base_name_b}_B_highlight.pdf",
                            pdf_b_highlight_bytes,
                            file_name=f"{base_name_b}_B_highlight.pdf",
                            mime="application/pdf",
                            key="dl_b_highlight"
                        )
                        
                        # Annotated PDFs
                        st.sidebar.markdown("---")
                        st.sidebar.markdown("**📝 Annotated PDFs (Editable)**")
                        
                        tmp_annot_a = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp_annot_a_path = tmp_annot_a.name
                        tmp_annot_a.close()
                        
                        tmp_annot_b = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp_annot_b_path = tmp_annot_b.name
                        tmp_annot_b.close()
                        
                        create_annotated_pdfs(
                            tmp_highlight_a_path,
                            tmp_highlight_b_path,
                            tmp_annot_a_path,
                            tmp_annot_b_path,
                            result['regions_a'],
                            result['regions_b'],
                            result['dimensions_a'],
                            result['dimensions_b'],
                            dpi=200
                        )
                        
                        num_regions_a = sum(len(r) for r in result['regions_a'])
                        num_regions_b = sum(len(r) for r in result['regions_b'])
                        
                        with open(tmp_annot_a_path, 'rb') as f:
                            pdf_annot_a_bytes = f.read()
                        with open(tmp_annot_b_path, 'rb') as f:
                            pdf_annot_b_bytes = f.read()
                        
                        st.sidebar.download_button(
                            f"💬 {base_name_a}_A_highlight_cloud.pdf ({num_regions_a} clouds)",
                            pdf_annot_a_bytes,
                            file_name=f"{base_name_a}_A_highlight_cloud.pdf",
                            mime="application/pdf",
                            key="dl_annot_a"
                        )
                        
                        st.sidebar.download_button(
                            f"💬 {base_name_b}_B_highlight_cloud.pdf ({num_regions_b} clouds)",
                            pdf_annot_b_bytes,
                            file_name=f"{base_name_b}_B_highlight_cloud.pdf",
                            mime="application/pdf",
                            key="dl_annot_b"
                        )
                        
                        # Cleanup
                        try:
                            os.unlink(tmp_highlight_a_path)
                            os.unlink(tmp_highlight_b_path)
                            os.unlink(tmp_annot_a_path)
                            os.unlink(tmp_annot_b_path)
                        except:
                            pass
        
        finally:
            pass
    
    else:
        st.info("👆 Please upload two PDF files to begin comparison")

else:
    # Batch mode - same logic as single pair but with multiple files
    pdf_a_files = st.sidebar.file_uploader(
        "PDF A (Original)", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="pdf_a_batch"
    )
    pdf_b_files = st.sidebar.file_uploader(
        "PDF B (Compare)", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="pdf_b_batch"
    )
    
    if pdf_a_files and pdf_b_files:
        # Clear previous results if files changed
        if 'batch_previous_files' not in st.session_state:
            st.session_state['batch_previous_files'] = None
        
        current_files = (tuple(f.name for f in pdf_a_files), tuple(f.name for f in pdf_b_files))
        if st.session_state['batch_previous_files'] != current_files:
            # Files changed, clear old results
            if 'batch_results' in st.session_state:
                del st.session_state['batch_results']
            if 'batch_zip_bytes' in st.session_state:
                del st.session_state['batch_zip_bytes']
            if 'batch_output_dir' in st.session_state:
                del st.session_state['batch_output_dir']
            st.session_state['batch_previous_files'] = current_files
        
        if len(pdf_a_files) != len(pdf_b_files):
            st.warning(f"⚠️ Warning: PDF A has {len(pdf_a_files)} file(s), PDF B has {len(pdf_b_files)} file(s). They will be paired sequentially.")
        
        num_pairs = min(len(pdf_a_files), len(pdf_b_files))
        st.info(f"📊 {num_pairs} PDF pair(s) will be processed")
        
        if st.sidebar.button("🚀 Process All Pairs", type="primary"):
            # Create temporary directory for outputs
            output_dir = tempfile.mkdtemp()
            all_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                for pair_idx in range(num_pairs):
                    pdf_a_file = pdf_a_files[pair_idx]
                    pdf_b_file = pdf_b_files[pair_idx]
                    
                    base_name_a = Path(pdf_a_file.name).stem
                    base_name_b = Path(pdf_b_file.name).stem
                    
                    status_text.text(f"Processing pair {pair_idx + 1}/{num_pairs}: {base_name_a} vs {base_name_b}")
                    
                    # Save files temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_a:
                        tmp_a.write(pdf_a_file.read())
                        tmp_a_path = tmp_a.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_b:
                        tmp_b.write(pdf_b_file.read())
                        tmp_b_path = tmp_b.name
                    
                    try:
                        # Process pair (same logic as single pair mode)
                        result = process_pdf_pair(
                            tmp_a_path, tmp_b_path,
                            base_name_a, base_name_b,
                            sensitivity
                        )
                        
                        if result:
                            # Create output files (use A/B labels like single pair mode)
                            output_files = create_output_files(result, output_dir, is_batch=False, compress=use_compression, quality=jpeg_quality)
                            all_results.append({
                                'pair_name': f"{base_name_a}_vs_{base_name_b}",
                                'base_name_a': base_name_a,
                                'base_name_b': base_name_b,
                                'files': output_files,
                                'num_regions_a': sum(len(r) for r in result['regions_a']),
                                'num_regions_b': sum(len(r) for r in result['regions_b'])
                            })
                        
                        # Cleanup temp PDF files
                        try:
                            os.unlink(tmp_a_path)
                            os.unlink(tmp_b_path)
                        except:
                            pass
                    
                    except Exception as e:
                        st.error(f"Error processing pair {pair_idx + 1}: {e}")
                        import traceback
                        st.text(traceback.format_exc())
                    
                    progress_bar.progress((pair_idx + 1) / num_pairs)
                
                status_text.text("✅ Processing complete!")
                
                # Create ZIP file with all results
                if all_results:
                    zip_path = os.path.join(output_dir, "batch_results.zip")
                    
                    # Verify files exist before creating ZIP
                    files_to_zip = []
                    for result in all_results:
                        for file_path in result['files']:
                            if os.path.exists(file_path):
                                arcname = os.path.join(result['pair_name'], os.path.basename(file_path))
                                files_to_zip.append((file_path, arcname))
                            else:
                                st.warning(f"File not found: {file_path}")
                    
                    if files_to_zip:
                        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            for file_path, arcname in files_to_zip:
                                zipf.write(file_path, arcname)
                        
                        # Verify ZIP was created
                        if not os.path.exists(zip_path):
                            st.error("Failed to create ZIP file")
                        else:
                            # Read ZIP file into memory and store in session state
                            with open(zip_path, 'rb') as f:
                                zip_bytes = f.read()
                            
                            if len(zip_bytes) == 0:
                                st.error("ZIP file is empty")
                            else:
                                # Store results in session state for persistent download
                                st.session_state['batch_results'] = all_results
                                st.session_state['batch_zip_bytes'] = zip_bytes
                                st.session_state['batch_output_dir'] = output_dir
                                # Update previous files to prevent clearing results
                                st.session_state['batch_previous_files'] = (tuple(f.name for f in pdf_a_files), tuple(f.name for f in pdf_b_files))
                                
                                # Display results summary
                                st.success(f"✅ Successfully processed {len(all_results)} pair(s). ZIP file size: {len(zip_bytes) / 1024:.1f} KB")
                    else:
                        st.error("No valid files to include in ZIP")
                else:
                    st.error("No results generated")
                    if 'batch_results' in st.session_state:
                        del st.session_state['batch_results']
                    if 'batch_zip_bytes' in st.session_state:
                        del st.session_state['batch_zip_bytes']
            
            except Exception as e:
                st.error(f"Error during batch processing: {e}")
                import traceback
                st.text(traceback.format_exc())
        
        # Display processing summary if results exist
        if 'batch_results' in st.session_state:
            st.markdown("### Processing Summary")
            for i, result in enumerate(st.session_state['batch_results'], 1):
                st.markdown(f"**Pair {i}: {result['base_name_a']} (PDF A) vs {result['base_name_b']} (PDF B)**")
                st.markdown(f"- PDF A: {result['num_regions_a']} cloud(s)")
                st.markdown(f"- PDF B: {result['num_regions_b']} cloud(s)")
                st.markdown(f"- Output files: {len(result['files'])}")
        
        # Display download buttons if results exist in session state
        if 'batch_results' in st.session_state:
            if 'batch_zip_bytes' not in st.session_state:
                st.error("⚠️ ZIP file data not found in session. Please process again.")
            else:
                zip_size = len(st.session_state['batch_zip_bytes'])
                if zip_size == 0:
                    st.error("⚠️ ZIP file is empty. Please process again.")
                else:
                    # Main area download section
                    st.markdown("---")
                    st.markdown("### 📥 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            "📦 Download All Results (ZIP)",
                            st.session_state['batch_zip_bytes'],
                            file_name="batch_results.zip",
                            mime="application/zip",
                            key="dl_batch_zip_main",
                            use_container_width=True
                        )
                    
                    with col2:
                        if st.button("🔄 Refresh Downloads", use_container_width=True):
                            st.rerun()
                    
                    # Sidebar download section
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**📦 Batch Download**")
                    
                    st.sidebar.download_button(
                        "📥 Download All Results (ZIP)",
                        st.session_state['batch_zip_bytes'],
                        file_name="batch_results.zip",
                        mime="application/zip",
                        key="dl_batch_zip_sidebar"
                    )
                    
                    # Individual file downloads in sidebar
                    st.sidebar.markdown("---")
                    st.sidebar.markdown("**📄 Individual Downloads**")
                    
                    for result_idx, result in enumerate(st.session_state['batch_results']):
                        with st.sidebar.expander(f"Pair {result_idx + 1}: {result['pair_name']}"):
                            for file_idx, file_path in enumerate(result['files']):
                                if os.path.exists(file_path):
                                    try:
                                        with open(file_path, 'rb') as f:
                                            file_bytes = f.read()
                                        
                                        if len(file_bytes) > 0:
                                            filename = os.path.basename(file_path)
                                            st.download_button(
                                                filename,
                                                file_bytes,
                                                file_name=filename,
                                                mime="application/pdf",
                                                key=f"dl_batch_{result_idx}_{file_idx}_{filename}"
                                            )
                                        else:
                                            st.warning(f"File is empty: {filename}")
                                    except Exception as e:
                                        st.error(f"Error reading file {file_path}: {e}")
                                else:
                                    st.warning(f"File not found: {file_path}")
    
    else:
        st.info("👆 Please upload PDF files in both PDF A and PDF B to begin batch processing")
