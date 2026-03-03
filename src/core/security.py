"""
Security utilities for PDF processing
"""
import os
import tempfile
import atexit
from pathlib import Path
from typing import BinaryIO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
MAX_PDF_SIZE_MB = int(os.getenv('MAX_PDF_SIZE_MB', '50'))
MAX_PDF_PAGES = int(os.getenv('MAX_PDF_PAGES', '100'))
PDF_TIMEOUT_SECONDS = int(os.getenv('PDF_PROCESSING_TIMEOUT_SECONDS', '300'))


class SecurityError(Exception):
    """Custom exception for security violations"""
    pass


def validate_pdf_file(uploaded_file: BinaryIO, max_size_mb: int = MAX_PDF_SIZE_MB) -> None:
    """
    Validate uploaded PDF file for security
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_size_mb: Maximum file size in megabytes
        
    Raises:
        SecurityError: If validation fails
    """
    # Reset file pointer to beginning
    uploaded_file.seek(0)
    
    # Check file size
    uploaded_file.seek(0, os.SEEK_END)
    size_bytes = uploaded_file.tell()
    size_mb = size_bytes / (1024 * 1024)
    uploaded_file.seek(0)
    
    if size_mb > max_size_mb:
        logger.warning(f"File size validation failed: {size_mb:.2f}MB > {max_size_mb}MB")
        raise SecurityError(
            f"檔案太大：{size_mb:.1f}MB（最大 {max_size_mb}MB）\n"
            f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)"
        )
    
    # Check PDF magic bytes (PDF header)
    header = uploaded_file.read(8)
    uploaded_file.seek(0)
    
    if not header.startswith(b'%PDF-'):
        logger.warning(f"Invalid PDF header: {header[:20]}")
        raise SecurityError(
            "無效的 PDF 檔案格式\n"
            "Invalid PDF file format"
        )
    
    logger.info(f"PDF validation passed: {size_mb:.2f}MB")


def validate_page_count(pdf_path: str, max_pages: int = MAX_PDF_PAGES) -> int:
    """
    Validate PDF page count
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum allowed pages
        
    Returns:
        Number of pages
        
    Raises:
        SecurityError: If page count exceeds limit
    """
    import fitz
    
    try:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
    except Exception as e:
        logger.error(f"Failed to open PDF: {e}")
        raise SecurityError(
            "無法讀取 PDF 檔案\n"
            "Failed to read PDF file"
        )
    
    if count > max_pages:
        logger.warning(f"Page count validation failed: {count} > {max_pages}")
        raise SecurityError(
            f"PDF 頁數過多：{count} 頁（最大 {max_pages} 頁）\n"
            f"Too many pages: {count} pages (max {max_pages} pages)"
        )
    
    logger.info(f"Page count validation passed: {count} pages")
    return count


def sanitize_filename(filename: str, max_length: int = 100) -> str:
    """
    Sanitize user-provided filename
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    import re
    
    # Get basename only (remove path)
    filename = Path(filename).name
    
    # Remove non-alphanumeric characters except - _ .
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Limit length
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[:max_length - len(ext)] + ext
    
    # Prevent hidden files
    if filename.startswith('.'):
        filename = '_' + filename[1:]
    
    # Ensure not empty
    if not filename:
        filename = 'output'
    
    return filename


class TempFileManager:
    """
    Temporary file manager with automatic cleanup
    Thread-safe for Streamlit session state
    """
    
    def __init__(self):
        self.temp_files = set()
        self.temp_dirs = set()
        atexit.register(self.cleanup_all)
    
    def create_temp_file(self, suffix: str = ".pdf", delete: bool = False) -> str:
        """
        Create temporary file
        
        Args:
            suffix: File suffix
            delete: If True, file auto-deletes on close
            
        Returns:
            Path to temporary file
        """
        tmp = tempfile.NamedTemporaryFile(delete=delete, suffix=suffix)
        tmp_path = tmp.name
        
        if not delete:
            self.temp_files.add(tmp_path)
            tmp.close()  # Close but don't delete
        
        logger.debug(f"Created temp file: {tmp_path}")
        return tmp_path
    
    def create_temp_dir(self) -> str:
        """
        Create temporary directory
        
        Returns:
            Path to temporary directory
        """
        tmp_dir = tempfile.mkdtemp()
        self.temp_dirs.add(tmp_dir)
        logger.debug(f"Created temp dir: {tmp_dir}")
        return tmp_dir
    
    def cleanup_file(self, filepath: str) -> None:
        """Clean up specific file"""
        try:
            Path(filepath).unlink(missing_ok=True)
            self.temp_files.discard(filepath)
            logger.debug(f"Cleaned up file: {filepath}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {filepath}: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all tracked temporary files and directories"""
        # Clean up files
        for filepath in list(self.temp_files):
            try:
                Path(filepath).unlink(missing_ok=True)
                logger.debug(f"Cleaned up file: {filepath}")
            except Exception as e:
                logger.warning(f"Failed to cleanup file {filepath}: {e}")
        
        # Clean up directories
        for dirpath in list(self.temp_dirs):
            try:
                import shutil
                shutil.rmtree(dirpath, ignore_errors=True)
                logger.debug(f"Cleaned up dir: {dirpath}")
            except Exception as e:
                logger.warning(f"Failed to cleanup dir {dirpath}: {e}")
        
        self.temp_files.clear()
        self.temp_dirs.clear()
        logger.info("All temporary files cleaned up")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup_all()


def safe_error_message(error: Exception, debug: bool = False) -> str:
    """
    Generate safe error message for users
    
    Args:
        error: Original exception
        debug: If True, include technical details
        
    Returns:
        User-friendly error message
    """
    # Log full error server-side
    logger.error(f"Error occurred: {type(error).__name__}: {error}", exc_info=True)
    
    # Generic user message
    user_message = (
        "處理過程中發生錯誤，請重試\n"
        "An error occurred during processing. Please try again."
    )
    
    # Add specific guidance for known error types
    if isinstance(error, SecurityError):
        user_message = str(error)
    elif "memory" in str(error).lower():
        user_message = (
            "記憶體不足，請嘗試較小的檔案\n"
            "Insufficient memory. Please try a smaller file."
        )
    elif "timeout" in str(error).lower():
        user_message = (
            "處理超時，請嘗試較小的檔案\n"
            "Processing timeout. Please try a smaller file."
        )
    
    # Add technical details if debug mode
    if debug:
        user_message += f"\n\nDebug: {type(error).__name__}: {error}"
    
    return user_message

