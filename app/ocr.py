"""
OCR (Optical Character Recognition) module for NotesGPT.
Handles text extraction from images using EasyOCR with multi-language support.
Includes image preprocessing for better OCR accuracy.
"""

import os
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging

# Image processing imports
try:
    import cv2
    import numpy as np
    from PIL import Image
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# EasyOCR import
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# PDF processing
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ['.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif', '.webp']

# Supported languages for OCR
# EasyOCR language codes: 'en' (English), 'fr' (French), 'ar' (Arabic), etc.
# Note: Arabic cannot be combined with French/Spanish/etc. Use ['en'] for English-only or ['en', 'ar'] for English+Arabic
DEFAULT_LANGUAGES = ['en', 'fr']  # English and French (most common for documents)

# OCR Reader instance (lazy-loaded)
_ocr_reader = None


# ==================== OCR Reader Initialization ====================

def get_ocr_reader(languages: List[str] = None) -> Optional[Any]:
    """
    Get or initialize the EasyOCR reader.
    
    Args:
        languages: List of language codes to support (e.g., ['en', 'fr', 'ar'])
                  If None, uses DEFAULT_LANGUAGES
    
    Returns:
        EasyOCR Reader instance or None if EasyOCR not available
    """
    global _ocr_reader
    
    if not EASYOCR_AVAILABLE:
        logger.error("EasyOCR is not installed. Please install it with: pip install easyocr")
        return None
    
    if languages is None:
        languages = DEFAULT_LANGUAGES
    
    # Initialize reader if not already done or if languages changed
    if _ocr_reader is None:
        try:
            logger.info(f"Initializing EasyOCR reader with languages: {languages}")
            _ocr_reader = easyocr.Reader(languages, gpu=False)  # Set gpu=True if you have CUDA
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            return None
    
    return _ocr_reader


# ==================== Image Preprocessing ====================

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy.
    Applies noise reduction, contrast enhancement, and adaptive thresholding.
    
    Args:
        image: Input image as numpy array (from cv2)
    
    Returns:
        Preprocessed image as numpy array
    """
    if not CV2_AVAILABLE:
        return image
    
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply noise reduction using bilateral filter
        # Preserves edges while removing noise
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding to handle varying lighting conditions
        # This creates a binary image (black text on white background)
        threshold = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        return threshold
    
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}. Using original image.")
        return image


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image file and convert it to numpy array.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as numpy array or None if failed
    """
    try:
        # Use PIL to load, then convert to cv2 format
        pil_image = Image.open(image_path)
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL image to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV format)
        if CV2_AVAILABLE:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Load an image from bytes (e.g., from uploaded file).
    
    Args:
        image_bytes: Image data as bytes
    
    Returns:
        Image as numpy array or None if failed
    """
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL image to numpy array
        image_array = np.array(pil_image)
        
        # Convert RGB to BGR (OpenCV format)
        if CV2_AVAILABLE:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {e}")
        return None


# ==================== OCR Extraction ====================

def extract_text_from_image(
    image: np.ndarray,
    languages: List[str] = None,
    preprocess: bool = True
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extract text from an image using EasyOCR.
    
    Args:
        image: Input image as numpy array
        languages: List of language codes (e.g., ['en', 'fr', 'ar'])
        preprocess: Whether to apply preprocessing for better accuracy
    
    Returns:
        Tuple of (extracted_text, detection_details)
        - extracted_text: Full text extracted from the image
        - detection_details: List of dictionaries with bounding boxes and confidence scores
    """
    if not EASYOCR_AVAILABLE:
        return "EasyOCR is not installed. Please install it with: pip install easyocr", []
    
    # Get OCR reader
    reader = get_ocr_reader(languages)
    if reader is None:
        return "Failed to initialize OCR reader", []
    
    try:
        # Preprocess image if requested
        if preprocess and CV2_AVAILABLE:
            processed_image = preprocess_image(image)
        else:
            processed_image = image
        
        # Perform OCR
        logger.info("Performing OCR text extraction...")
        results = reader.readtext(processed_image)
        
        # Extract text and details
        extracted_lines = []
        detection_details = []
        
        for detection in results:
            bbox, text, confidence = detection
            extracted_lines.append(text)
            
            detection_details.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": bbox
            })
        
        # Combine all text lines
        full_text = "\n".join(extracted_lines)
        
        logger.info(f"OCR completed. Extracted {len(extracted_lines)} text segments.")
        
        return full_text, detection_details
    
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return f"Error during OCR: {str(e)}", []


def extract_text_from_image_file(
    image_path: str,
    languages: List[str] = None,
    preprocess: bool = True
) -> Dict[str, Any]:
    """
    Extract text from an image file with comprehensive error handling.
    
    Args:
        image_path: Path to the image file
        languages: List of language codes (e.g., ['en', 'fr', 'ar'])
        preprocess: Whether to apply preprocessing for better accuracy
    
    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - text: Extracted text
        - details: Detection details with confidence scores
        - error: Error message (if failed)
    """
    # Check if file exists
    if not Path(image_path).exists():
        return {
            "success": False,
            "text": "",
            "details": [],
            "error": f"File not found: {image_path}"
        }
    
    # Check file extension
    file_ext = Path(image_path).suffix.lower()
    if file_ext not in SUPPORTED_IMAGE_FORMATS:
        return {
            "success": False,
            "text": "",
            "details": [],
            "error": f"Unsupported image format: {file_ext}. Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
        }
    
    # Load image
    image = load_image(image_path)
    if image is None:
        return {
            "success": False,
            "text": "",
            "details": [],
            "error": "Failed to load image"
        }
    
    # Extract text
    text, details = extract_text_from_image(image, languages, preprocess)
    
    # Check if extraction was successful
    if not text or text.startswith("Error") or text.startswith("EasyOCR"):
        return {
            "success": False,
            "text": text,
            "details": details,
            "error": text
        }
    
    return {
        "success": True,
        "text": text,
        "details": details,
        "error": None
    }


def extract_text_from_image_bytes(
    image_bytes: bytes,
    filename: str = "image",
    languages: List[str] = None,
    preprocess: bool = True
) -> Dict[str, Any]:
    """
    Extract text from image bytes (for uploaded files).
    
    Args:
        image_bytes: Image data as bytes
        filename: Name of the file (for logging/error messages)
        languages: List of language codes (e.g., ['en', 'fr', 'ar'])
        preprocess: Whether to apply preprocessing for better accuracy
    
    Returns:
        Dictionary containing:
        - success: Boolean indicating success
        - text: Extracted text
        - details: Detection details with confidence scores
        - error: Error message (if failed)
    """
    # Load image from bytes
    image = load_image_from_bytes(image_bytes)
    if image is None:
        return {
            "success": False,
            "text": "",
            "details": [],
            "error": f"Failed to load image: {filename}"
        }
    
    # Extract text
    text, details = extract_text_from_image(image, languages, preprocess)
    
    # Check if extraction was successful
    if not text or text.startswith("Error") or text.startswith("EasyOCR"):
        return {
            "success": False,
            "text": text,
            "details": details,
            "error": text
        }
    
    return {
        "success": True,
        "text": text,
        "details": details,
        "error": None
    }


# ==================== PDF OCR Support ====================

def extract_text_from_pdf_with_ocr(
    pdf_path: str,
    languages: List[str] = None,
    preprocess: bool = True
) -> Dict[str, Any]:
    """
    Extract text from a scanned PDF using OCR.
    Converts each PDF page to an image and performs OCR.
    
    Args:
        pdf_path: Path to the PDF file
        languages: List of language codes
        preprocess: Whether to apply preprocessing
    
    Returns:
        Dictionary with extracted text per page
    """
    if not PDF2IMAGE_AVAILABLE:
        return {
            "success": False,
            "text": "",
            "pages": [],
            "error": "pdf2image is not installed. Please install it with: pip install pdf2image"
        }
    
    try:
        # Convert PDF pages to images
        logger.info(f"Converting PDF to images: {pdf_path}")
        images = convert_from_path(pdf_path)
        
        pages_text = []
        all_text_lines = []
        
        for page_num, image in enumerate(images, start=1):
            logger.info(f"Processing page {page_num}/{len(images)}")
            
            # Convert PIL image to numpy array
            image_array = np.array(image)
            if CV2_AVAILABLE:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Extract text from this page
            page_text, details = extract_text_from_image(image_array, languages, preprocess)
            
            pages_text.append({
                "page": page_num,
                "text": page_text,
                "details": details
            })
            
            all_text_lines.append(f"[Page {page_num}]\n{page_text}\n")
        
        full_text = "\n".join(all_text_lines)
        
        return {
            "success": True,
            "text": full_text,
            "pages": pages_text,
            "total_pages": len(images),
            "error": None
        }
    
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        return {
            "success": False,
            "text": "",
            "pages": [],
            "error": f"Failed to process PDF: {str(e)}"
        }


# ==================== Utility Functions ====================

def is_ocr_available() -> bool:
    """Check if all required OCR dependencies are installed."""
    return EASYOCR_AVAILABLE and CV2_AVAILABLE


def get_supported_languages() -> List[str]:
    """Get list of supported OCR languages."""
    if not EASYOCR_AVAILABLE:
        return []
    
    # EasyOCR supports many languages
    # Here we return the most common ones
    return [
        'en',  # English
        'fr',  # French
        'ar',  # Arabic
        'es',  # Spanish
        'de',  # German
        'it',  # Italian
        'pt',  # Portuguese
        'ru',  # Russian
        'zh',  # Chinese (Simplified)
        'ja',  # Japanese
        'ko',  # Korean
    ]


def get_ocr_info() -> Dict[str, Any]:
    """Get information about OCR capabilities."""
    return {
        "available": is_ocr_available(),
        "engine": "EasyOCR" if EASYOCR_AVAILABLE else None,
        "preprocessing": CV2_AVAILABLE,
        "pdf_support": PDF2IMAGE_AVAILABLE,
        "default_languages": DEFAULT_LANGUAGES,
        "supported_languages": get_supported_languages(),
        "supported_formats": SUPPORTED_IMAGE_FORMATS
    }
