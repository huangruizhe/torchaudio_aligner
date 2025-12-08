"""
Text Loading Utilities

Functions for loading text from various sources:
- Local files (TXT)
- URLs (HTML)
- PDFs (text extraction and OCR)
"""

from pathlib import Path
from typing import Union, List, Optional
import logging

logger = logging.getLogger(__name__)

# Optional dependencies
_REQUESTS_AVAILABLE = False
_BS4_AVAILABLE = False
_PYPDF_AVAILABLE = False
_OCR_AVAILABLE = False
_PDF2IMAGE_AVAILABLE = False

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    BeautifulSoup = None

try:
    from pypdf import PdfReader
    _PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None

try:
    import easyocr
    _OCR_AVAILABLE = True
except ImportError:
    easyocr = None

try:
    from pdf2image import convert_from_path
    _PDF2IMAGE_AVAILABLE = True
except ImportError:
    convert_from_path = None


def load_text_from_file(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    """
    Load text from a local file.

    Args:
        file_path: Path to text file
        encoding: File encoding (default utf-8)

    Returns:
        Text content as string
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")

    logger.info(f"Loading text from file: {file_path}")
    with open(file_path, "r", encoding=encoding) as f:
        text = f.read()

    logger.info(f"Loaded {len(text)} characters, {len(text.split())} words")
    return text


def load_text_from_url(url: str, timeout: int = 30, encoding: Optional[str] = None) -> str:
    """
    Load text from a URL (HTML page).

    Requires: pip install requests beautifulsoup4

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds
        encoding: Force specific encoding. If None, tries auto-detection with
            fallback to Japanese encodings (shiftjis, shift_jisx0213) for
            compatibility with Japanese websites.

    Returns:
        Extracted text content
    """
    if not _REQUESTS_AVAILABLE:
        raise ImportError("requests is required. Install with: pip install requests")
    if not _BS4_AVAILABLE:
        raise ImportError("beautifulsoup4 is required. Install with: pip install beautifulsoup4")

    import html

    logger.info(f"Fetching text from URL: {url}")
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    # Decode response with encoding fallbacks
    if encoding:
        text_content = response.content.decode(encoding)
    else:
        # Try standard decoding first, then Japanese encodings as fallback
        encodings_to_try = ['utf-8', 'shiftjis', 'shift_jisx0213', 'euc-jp', 'iso-2022-jp']
        text_content = None
        for enc in encodings_to_try:
            try:
                text_content = response.content.decode(enc)
                logger.debug(f"Successfully decoded with encoding: {enc}")
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if text_content is None:
            # Fall back to requests' detected encoding
            text_content = response.text

    # Unescape HTML entities
    text_content = html.unescape(text_content)

    soup = BeautifulSoup(text_content, "html.parser")
    text = soup.get_text()
    text = text.replace("\r\n", "\n")

    logger.info(f"Loaded {len(text)} characters, {len(text.split())} words")
    return text


def load_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    """
    Extract text from a PDF file.

    Requires: pip install pypdf

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text content
    """
    if not _PYPDF_AVAILABLE:
        raise ImportError("pypdf is required. Install with: pip install pypdf")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    logger.info(f"Extracting text from PDF: {pdf_path}")
    reader = PdfReader(str(pdf_path))

    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text())
    text = " ".join(text_parts)

    logger.info(f"Extracted {len(text)} characters, {len(text.split())} words from {len(reader.pages)} pages")
    return text


def load_text_from_pdf_ocr(
    pdf_path: Union[str, Path],
    languages: List[str] = None,
    dpi: int = 300,
    fallback_to_text: bool = True,
) -> str:
    """
    Extract text from a scanned/image-based PDF using OCR.

    This is useful for PDFs that contain scanned images instead of text
    (common for Hindi, historical documents, etc.).

    Requires: pip install easyocr pdf2image
    Also requires poppler-utils system package for pdf2image.

    Args:
        pdf_path: Path to PDF file
        languages: List of language codes for EasyOCR (e.g., ['en'], ['hi', 'en'])
            Default: ['en']
        dpi: Resolution for PDF to image conversion (higher = better quality but slower)
        fallback_to_text: If True, try text extraction first, use OCR only if empty

    Returns:
        Extracted text content

    Example:
        >>> # For Hindi scanned PDF
        >>> text = load_text_from_pdf_ocr("hindi_document.pdf", languages=['hi', 'en'])

        >>> # For English scanned PDF
        >>> text = load_text_from_pdf_ocr("scanned_book.pdf", languages=['en'])
    """
    if not _PDF2IMAGE_AVAILABLE:
        raise ImportError(
            "pdf2image is required for OCR. Install with: pip install pdf2image\n"
            "Also install poppler-utils: brew install poppler (macOS) or apt install poppler-utils (Linux)"
        )
    if not _OCR_AVAILABLE:
        raise ImportError("easyocr is required for OCR. Install with: pip install easyocr")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    languages = languages or ['en']

    # Try text extraction first if fallback enabled
    if fallback_to_text:
        try:
            text = load_text_from_pdf(pdf_path)
            # Check if we got meaningful text (not just whitespace/noise)
            if len(text.strip()) > 100 and len(text.split()) > 20:
                logger.info("PDF contains extractable text, skipping OCR")
                return text
        except Exception:
            pass
        logger.info("PDF appears to be scanned/image-based, using OCR")

    logger.info(f"Converting PDF to images (dpi={dpi}): {pdf_path}")
    images = convert_from_path(str(pdf_path), dpi=dpi)
    logger.info(f"Converted {len(images)} pages to images")

    # Initialize EasyOCR reader
    logger.info(f"Initializing EasyOCR with languages: {languages}")
    reader = easyocr.Reader(languages, gpu=False)  # CPU by default for compatibility

    text_parts = []
    for i, image in enumerate(images):
        logger.info(f"OCR processing page {i+1}/{len(images)}")
        # EasyOCR expects numpy array or file path
        import numpy as np
        image_np = np.array(image)
        results = reader.readtext(image_np, detail=0)  # detail=0 returns text only
        page_text = " ".join(results)
        text_parts.append(page_text)

    text = " ".join(text_parts)
    logger.info(f"OCR extracted {len(text)} characters, {len(text.split())} words from {len(images)} pages")
    return text


def load_text(source: Union[str, Path]) -> str:
    """
    Load text from any source (auto-detect type).

    Convenience function that automatically detects the source type:
    - URLs (http/https): Uses load_text_from_url
    - PDF files (.pdf): Uses load_text_from_pdf
    - Other files: Uses load_text_from_file

    Args:
        source: File path or URL

    Returns:
        Extracted text content

    Example:
        >>> text = load_text("transcript.pdf")
        >>> text = load_text("https://example.com/page.html")
        >>> text = load_text("transcript.txt")
    """
    s = str(source)
    if s.startswith("http://") or s.startswith("https://"):
        return load_text_from_url(s)
    if s.lower().endswith(".pdf"):
        return load_text_from_pdf(source)
    return load_text_from_file(source)


def get_available_loaders() -> dict:
    """Return availability of text loading backends."""
    return {
        "file": True,
        "url": _REQUESTS_AVAILABLE and _BS4_AVAILABLE,
        "pdf": _PYPDF_AVAILABLE,
        "ocr": _OCR_AVAILABLE and _PDF2IMAGE_AVAILABLE,
    }
