#Handles images (PNG, JPG) using Tesseract.
import logging
from PIL import Image
import pytesseract

# Configure basic logging to catch silent failures
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image using Tesseract OCR.
    Fails gracefully to prevent the entire app from crashing on a bad image.
    """
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        cleaned_text = text.strip()
        
        if not cleaned_text:
            logger.warning(f"OCR completed but found no text in: {image_path}")
            
        return cleaned_text
    except Exception as e:
        logger.error(f"OCR Engine Failure for {image_path}. Error: {e}")
        return ""