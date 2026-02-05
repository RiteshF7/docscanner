"""
PDF Maker - PDF Creation Logic
------------------------------
Handles combining images into PDF documents.
"""

import img2pdf
from pathlib import Path
from PIL import Image
import io


def images_to_pdf(image_paths, output_path):
    """
    Combine multiple images into a single PDF.
    
    Args:
        image_paths: List of image file paths
        output_path: Path for output PDF
    
    Returns:
        Path to created PDF
    """
    if not image_paths:
        raise ValueError("No images provided")
    
    # Convert all paths to strings and verify they exist
    valid_paths = []
    for path in image_paths:
        path = Path(path)
        if path.exists():
            valid_paths.append(str(path))
        else:
            print(f"Warning: Image not found: {path}")
    
    if not valid_paths:
        raise ValueError("No valid images found")
    
    # Create PDF
    with open(output_path, 'wb') as f:
        f.write(img2pdf.convert(valid_paths))
    
    return output_path


def image_bytes_to_pdf(image_bytes_list, output_path):
    """
    Combine multiple image byte arrays into a single PDF.
    
    Args:
        image_bytes_list: List of image bytes
        output_path: Path for output PDF
    
    Returns:
        Path to created PDF
    """
    if not image_bytes_list:
        raise ValueError("No images provided")
    
    # Create PDF from bytes
    with open(output_path, 'wb') as f:
        f.write(img2pdf.convert(image_bytes_list))
    
    return output_path


def pil_images_to_pdf(pil_images, output_path, quality=95):
    """
    Combine multiple PIL images into a single PDF.
    
    Args:
        pil_images: List of PIL Image objects
        output_path: Path for output PDF
        quality: JPEG quality (1-100)
    
    Returns:
        Path to created PDF
    """
    if not pil_images:
        raise ValueError("No images provided")
    
    # Convert PIL images to bytes
    image_bytes_list = []
    for img in pil_images:
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        image_bytes_list.append(buffer.getvalue())
    
    return image_bytes_to_pdf(image_bytes_list, output_path)


def get_pdf_page_count(pdf_path):
    """
    Get the number of pages in a PDF.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Number of pages
    """
    try:
        import pikepdf
        with pikepdf.open(pdf_path) as pdf:
            return len(pdf.pages)
    except ImportError:
        # Fallback: count by file structure (rough estimate)
        return -1


if __name__ == "__main__":
    print("PDF Maker module loaded successfully!")
    print("Use images_to_pdf() to create PDFs from image files.")
