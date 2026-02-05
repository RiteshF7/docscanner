"""
Document Scanner - Core Transform Logic
---------------------------------------
Handles perspective transformation and image enhancement.
"""

import cv2
import numpy as np
from pathlib import Path


def order_points(pts):
    """
    Order points: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts, dtype="float32")
    
    # Sum: top-left smallest, bottom-right largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Diff: top-right smallest, bottom-left largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def four_point_transform(image, pts):
    """
    Apply perspective transform to get a top-down view of the document.
    
    Args:
        image: Input image (numpy array)
        pts: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    Returns:
        Transformed/warped image
    """
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    
    # Calculate dimensions
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Ensure minimum dimensions
    maxWidth = max(maxWidth, 100)
    maxHeight = max(maxHeight, 100)
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Apply transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped


def enhance_document(image, brightness=0, contrast=0):
    """
    Enhance the scanned document for better readability.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (-100 to 100)
    
    Returns:
        Enhanced image
    """
    # Convert to LAB for better contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply brightness and contrast adjustments
    if brightness != 0 or contrast != 0:
        beta = brightness  # Brightness
        alpha = 1 + (contrast / 100)  # Contrast
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    
    return enhanced


def add_white_border(image, padding=20):
    """
    Add a clean white border around the document.
    """
    return cv2.copyMakeBorder(
        image, 
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]
    )


def process_document(image_path, corners, output_path=None, enhance=True, add_border=True):
    """
    Process a document image with given corner points.
    
    Args:
        image_path: Path to input image
        corners: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        output_path: Path to save output (optional)
        enhance: Whether to enhance the document
        add_border: Whether to add white border
    
    Returns:
        Processed image (numpy array)
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Apply perspective transform
    warped = four_point_transform(image, corners)
    
    # Enhance if requested
    if enhance:
        warped = enhance_document(warped)
    
    # Add border if requested
    if add_border:
        warped = add_white_border(warped)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(str(output_path), warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return warped


def process_document_from_array(image_array, corners, enhance=True, add_border=True):
    """
    Process a document from numpy array with given corner points.
    
    Args:
        image_array: Input image as numpy array
        corners: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        enhance: Whether to enhance the document
        add_border: Whether to add white border
    
    Returns:
        Processed image (numpy array)
    """
    # Apply perspective transform
    warped = four_point_transform(image_array, corners)
    
    # Enhance if requested
    if enhance:
        warped = enhance_document(warped)
    
    # Add border if requested
    if add_border:
        warped = add_white_border(warped)
    
    return warped


if __name__ == "__main__":
    # Test with sample corners
    print("Document Scanner module loaded successfully!")
    print("Use process_document() or process_document_from_array() to scan documents.")
