"""
Document Scanner - Flask Server
-------------------------------
Web server for the document scanner application.

Usage:
    python server.py

Then open http://localhost:5050 in your browser.
"""

from flask import Flask, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import base64
import uuid
from pathlib import Path
from datetime import datetime

# Import local modules
from scanner import process_document_from_array, four_point_transform, enhance_document, add_white_border
from pdf_maker import images_to_pdf

# Get the directory where server.py is located
BASE_DIR = Path(__file__).parent.resolve()

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = BASE_DIR / 'input'
OUTPUT_FOLDER = BASE_DIR / 'output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp', 'tiff'}

# Ensure folders exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Store scanned images in memory for PDF creation
scanned_images = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main HTML page"""
    html_path = BASE_DIR / 'index.html'
    with open(html_path, 'r', encoding='utf-8') as f:
        return Response(f.read(), mimetype='text/html')


@app.route('/health')
def health():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy'})


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Generate unique filename
        ext = file.filename.rsplit('.', 1)[1].lower()
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}.{ext}"
        filepath = UPLOAD_FOLDER / filename
        
        # Save file
        file.save(str(filepath))
        
        # Read image and get dimensions
        image = cv2.imread(str(filepath))
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400
        
        height, width = image.shape[:2]
        
        # Convert to base64 for preview
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': str(filepath),
            'width': width,
            'height': height,
            'preview': f'data:image/jpeg;base64,{img_base64}'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/scan', methods=['POST'])
def scan_document():
    """Process document with selected corners"""
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    filename = data.get('filename')
    corners = data.get('corners')  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    
    if not filename or not corners:
        return jsonify({'error': 'Missing filename or corners'}), 400
    
    if len(corners) != 4:
        return jsonify({'error': 'Exactly 4 corners required'}), 400
    
    # Read the image
    filepath = UPLOAD_FOLDER / filename
    image = cv2.imread(str(filepath))
    
    if image is None:
        return jsonify({'error': 'Could not read image'}), 400
    
    try:
        # Process the document
        scanned = process_document_from_array(
            image, 
            corners,
            enhance=data.get('enhance', True),
            add_border=data.get('addBorder', True)
        )
        
        # Generate output filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"scanned_{timestamp}_{filename}"
        output_path = OUTPUT_FOLDER / output_filename
        
        # Save scanned image
        cv2.imwrite(str(output_path), scanned, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # Store in memory for PDF creation
        scan_id = str(uuid.uuid4())[:8]
        scanned_images[scan_id] = {
            'path': str(output_path),
            'filename': output_filename,
            'timestamp': timestamp,
            'input_file': str(filepath)  # Track input file for cleanup
        }
        
        # Convert to base64 for preview
        _, buffer = cv2.imencode('.jpg', scanned, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'scan_id': scan_id,
            'filename': output_filename,
            'preview': f'data:image/jpeg;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/create-pdf', methods=['POST'])
def create_pdf():
    """Create PDF from scanned images"""
    data = request.json
    scan_ids = data.get('scan_ids', [])
    cleanup = data.get('cleanup', True)  # Cleanup by default
    
    if not scan_ids:
        # Use all scanned images
        scan_ids = list(scanned_images.keys())
    
    if not scan_ids:
        return jsonify({'error': 'No scanned images available'}), 400
    
    # Collect image paths and input files to clean up
    image_paths = []
    input_files_to_cleanup = set()
    
    for scan_id in scan_ids:
        if scan_id in scanned_images:
            image_paths.append(scanned_images[scan_id]['path'])
            # Track the original input file if it exists
            if 'input_file' in scanned_images[scan_id]:
                input_files_to_cleanup.add(scanned_images[scan_id]['input_file'])
    
    if not image_paths:
        return jsonify({'error': 'No valid images found'}), 400
    
    try:
        # Generate PDF filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_filename = f"document_{timestamp}.pdf"
        pdf_path = OUTPUT_FOLDER / pdf_filename
        
        # Create PDF
        images_to_pdf(image_paths, str(pdf_path))
        
        # Cleanup temporary files if requested
        cleaned_files = 0
        if cleanup:
            # Clean up scanned images (they're now in the PDF)
            for img_path in image_paths:
                try:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        cleaned_files += 1
                except Exception as e:
                    print(f"Warning: Could not delete {img_path}: {e}")
            
            # Clean up input/uploaded files
            for input_file in input_files_to_cleanup:
                try:
                    if os.path.exists(input_file):
                        os.remove(input_file)
                        cleaned_files += 1
                except Exception as e:
                    print(f"Warning: Could not delete {input_file}: {e}")
            
            # Clear the scanned images from memory
            for scan_id in scan_ids:
                if scan_id in scanned_images:
                    del scanned_images[scan_id]
        
        return jsonify({
            'success': True,
            'filename': pdf_filename,
            'path': str(pdf_path),
            'page_count': len(image_paths),
            'cleaned_files': cleaned_files
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """Download a file from output folder"""
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)


@app.route('/preview/<filename>')
def preview_file(filename):
    """Preview a file from output folder"""
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route('/list-scanned')
def list_scanned():
    """List all scanned images"""
    return jsonify({
        'images': [
            {
                'id': scan_id,
                'filename': info['filename'],
                'timestamp': info['timestamp']
            }
            for scan_id, info in scanned_images.items()
        ]
    })


@app.route('/clear-scanned', methods=['POST'])
def clear_scanned():
    """Clear scanned images from memory"""
    scanned_images.clear()
    return jsonify({'success': True})


@app.route('/reset-session', methods=['POST'])
def reset_session():
    """Reset the entire session - clear all files and memory"""
    cleaned = 0
    
    # Clean input folder
    for f in UPLOAD_FOLDER.iterdir():
        if f.is_file():
            try:
                os.remove(f)
                cleaned += 1
            except Exception as e:
                print(f"Warning: Could not delete {f}: {e}")
    
    # Clean output folder (including PDFs for full reset)
    for f in OUTPUT_FOLDER.iterdir():
        if f.is_file():
            try:
                os.remove(f)
                cleaned += 1
            except Exception as e:
                print(f"Warning: Could not delete {f}: {e}")
    
    # Clear memory
    scanned_images.clear()
    
    return jsonify({'success': True, 'cleaned_files': cleaned})


@app.route('/cleanup-all', methods=['POST'])
def cleanup_all():
    """Clean up all temporary files in input and output folders (except PDFs)"""
    cleaned = 0
    
    # Clean input folder
    for f in UPLOAD_FOLDER.iterdir():
        if f.is_file():
            try:
                os.remove(f)
                cleaned += 1
            except Exception as e:
                print(f"Warning: Could not delete {f}: {e}")
    
    # Clean scanned images in output folder (keep PDFs)
    for f in OUTPUT_FOLDER.iterdir():
        if f.is_file() and f.suffix.lower() != '.pdf':
            try:
                os.remove(f)
                cleaned += 1
            except Exception as e:
                print(f"Warning: Could not delete {f}: {e}")
    
    # Clear memory
    scanned_images.clear()
    
    return jsonify({'success': True, 'cleaned_files': cleaned})


@app.route('/input/<filename>')
def serve_input(filename):
    """Serve uploaded input images"""
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    print("=" * 50)
    print("Document Scanner Server")
    print("=" * 50)
    print(f"\nInput folder: {UPLOAD_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    
    # Use PORT from environment (for Render) or default to 5050
    port = int(os.environ.get('PORT', 5050))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"\nOpen http://localhost:{port} in your browser")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
