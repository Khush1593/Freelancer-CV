"""
Robust Dutch B-VCA Diploma Extractor
Enhanced with PDF to Image conversion and intelligent rotation detection

Strategy:
1. Try Unstructured API first (fast, works 75% of time) - Works on PDF directly
2. If fails, try Tesseract OCR on original image
3. If still fails AND it's a PDF: Convert PDF to images first
4. Apply intelligent rotation detection (0°, 90°, 180°, 270°) using OCR confidence
5. Apply image preprocessing (shadow removal, rotation correction) on converted images
6. Run Tesseract OCR on preprocessed images
"""

import re
import os
from typing import Optional, Dict, List, Tuple
import pytesseract
from PIL import Image
import cv2
import numpy as np
import unstructured_client
from unstructured_client.models import shared
from unstructured_client.models.operations import PartitionRequest
from dotenv import load_dotenv
import mimetypes
import fitz  # PyMuPDF for PDF to image conversion

load_dotenv()

def get_mime_type(file_path: str) -> str:
    """Get MIME type of file"""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def is_pdf(file_path: str) -> bool:
    """Check if file is a PDF"""
    return file_path.lower().endswith('.pdf')

def pdf_to_images(pdf_path: str, output_folder: str = "temp_pdf_images", dpi: int = 300) -> List[str]:
    """
    Convert PDF to high-quality images using PyMuPDF
    
    Args:
        pdf_path: Path to PDF file
        output_folder: Where to save images
        dpi: Resolution (300 is good for OCR)
    
    Returns:
        List of image paths created
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []
    
    try:
        print(f"\n📄 Converting PDF to images: {os.path.basename(pdf_path)}")
        
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        print(f"   Pages: {total_pages}, DPI: {dpi}")
        
        # Calculate zoom factor for desired DPI
        zoom = dpi / 72  # 72 is default DPI
        mat = fitz.Matrix(zoom, zoom)
        
        # Convert each page
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
            # Render page to image
            pix = page.get_pixmap(matrix=mat)
            
            # Generate filename
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            image_path = os.path.join(output_folder, f'{base_name}_page_{page_num + 1}.png')
            
            # Save
            pix.save(image_path)
            image_paths.append(image_path)
            
            print(f"   ✓ Page {page_num + 1}/{total_pages}: {os.path.basename(image_path)} "
                  f"({pix.width}x{pix.height})")
        
        pdf_document.close()
        print(f"✅ Successfully converted {total_pages} pages!\n")
        return image_paths
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return []

def get_ocr_confidence(image: np.ndarray, lang: str = 'eng+nld') -> float:
    """
    Get OCR confidence score for an image
    Higher confidence = better text recognition = correct orientation
    
    Args:
        image: Input image (numpy array)
        lang: Tesseract language
    
    Returns:
        Average confidence score (0-100)
    """
    try:
        # Get detailed OCR data including confidence
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        
        # Filter out empty text and get confidence scores
        confidences = [int(conf) for conf, text in zip(data['conf'], data['text']) 
                      if text.strip() and int(conf) > 0]
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            return avg_confidence
        return 0
        
    except Exception as e:
        print(f"  ⚠ Confidence calculation error: {e}")
        return 0

def detect_correct_orientation(image_path: str, output_folder: str = "output") -> Tuple[np.ndarray, int]:
    """
    Detect correct orientation by trying all 4 rotations (0°, 90°, 180°, 270°)
    and selecting the one with highest OCR confidence
    
    Args:
        image_path: Path to image file
        output_folder: Folder to save rotated images
    
    Returns:
        Tuple of (corrected_image, rotation_angle)
    """
    print("\n🔄 Detecting correct orientation...")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"  ✗ Error: Could not read image")
        return img, 0
    
    # Try all 4 rotations
    rotations = {
        0: img,  # Original
        90: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(img, cv2.ROTATE_180),
        270: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }
    
    # Calculate confidence for each rotation
    confidences = {}
    print("  Testing orientations:")
    
    for angle, rotated_img in rotations.items():
        # Convert to grayscale for OCR
        gray = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2GRAY)
        
        # Get confidence score
        confidence = get_ocr_confidence(gray)
        confidences[angle] = confidence
        
        print(f"    {angle:3d}° → Confidence: {confidence:.2f}%")
    
    # Find best orientation
    best_angle = max(confidences, key=confidences.get)
    best_confidence = confidences[best_angle]
    
    print(f"  ✓ Best orientation: {best_angle}° (confidence: {best_confidence:.2f}%)")
    
    # Get the best rotated image
    corrected_img = rotations[best_angle]
    
    # Save corrected image if rotation was needed
    if best_angle != 0:
        os.makedirs(output_folder, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_folder, f"{base_name}_rotated_{best_angle}.png")
        cv2.imwrite(output_path, corrected_img)
        print(f"  ✓ Saved corrected image: {os.path.basename(output_path)}")
    
    return corrected_img, best_angle

def detect_and_correct_fine_rotation(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct small rotation angles (±5°) using Hough line detection
    This is for fine-tuning after major rotation correction
    
    Args:
        image: Input image (BGR or grayscale)
    
    Returns:
        Rotated image with corrected orientation
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is not None:
            # Calculate average angle
            angles = []
            for line in lines[:20]:  # Use first 20 lines
                rho, theta = line[0]
                angle = np.degrees(theta) - 90
                
                # Filter out near-vertical lines
                if -5 < angle < 5:  # Only small corrections
                    angles.append(angle)
            
            if angles:
                # Get median angle
                median_angle = np.median(angles)
                
                # Only rotate if angle is significant (> 0.5 degrees)
                if abs(median_angle) > 0.5:
                    print(f"  - Fine rotation adjustment: {median_angle:.2f}°")
                    
                    # Get image dimensions
                    h, w = image.shape[:2]
                    center = (w // 2, h // 2)
                    
                    # Create rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                    
                    # Rotate image
                    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                            flags=cv2.INTER_CUBIC,
                                            borderMode=cv2.BORDER_REPLICATE)
                    
                    return rotated
        
        return image
        
    except Exception as e:
        print(f"  ⚠ Fine rotation correction failed: {e}")
        return image

def extract_with_unstructured(file_path: str) -> Optional[str]:
    """
    Method 1: Extract text using Unstructured API (FAST)
    Returns extracted text or None if failed
    """
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        print("⚠ Warning: UNSTRUCTURED_API_KEY not found, skipping Unstructured API")
        return None
    
    try:
        print("\n[Method 1/4] Trying Unstructured API...")
        client = unstructured_client.UnstructuredClient(api_key_auth=api_key)  
        
        with open(file_path, "rb") as f:
            req = PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=f.read(),
                        file_name=os.path.basename(file_path)
                    ),
                    strategy=shared.Strategy.HI_RES,
                    languages=["nld", "eng"],
                    ocr_languages=["nld", "eng"],
                )
            )
            response = client.general.partition(request=req)
        
        if hasattr(response, 'elements') and response.elements:
            text = "\n".join([element.get("text", "") for element in response.elements])
            if text and len(text.strip()) > 20:
                print(f"✓ Unstructured API extracted {len(text)} characters")
                return text
        
        print("⚠ Unstructured API returned insufficient text")
        return None
        
    except Exception as e:
        print(f"⚠ Unstructured API failed: {type(e).__name__}: {str(e)[:100]}")
        return None

def preprocess_image_advanced(file_path: str, output_folder: str = "output") -> dict:
    """
    Advanced image preprocessing with rotation correction, shadow removal and contrast enhancement
    
    Returns dict of paths to the preprocessed images
    """
    print("\n[Method 4/4] Using Advanced Image Preprocessing...")
    img = cv2.imread(file_path)
    
    if img is None:
        print(f"✗ Error: Could not read image file: {file_path}")
        return {}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get base filename
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Step 1: Major rotation correction (0°, 90°, 180°, 270°)
    print("  - Detecting major orientation...")
    corrected_img, rotation_angle = detect_correct_orientation(file_path, output_folder)
    
    # Step 2: Fine rotation correction (±5°)
    if corrected_img is not None:
        print("  - Fine-tuning rotation...")
        corrected_img = detect_and_correct_fine_rotation(corrected_img)
    else:
        corrected_img = img
    
    # Convert to grayscale
    gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Shadow removal using background estimation
    print("  - Removing shadows...")
    blur = cv2.GaussianBlur(gray, (55, 55), 0)
    no_shadow = cv2.divide(gray, blur, scale=255)
    
    # Step 4: Apply CLAHE for local contrast enhancement
    print("  - Enhancing contrast...")
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(no_shadow)
    
    # Step 5: Adaptive thresholding
    print("  - Applying adaptive threshold...")
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        9
    )
    
    # Save all preprocessed versions
    output_paths = {
        'rotated': os.path.join(output_folder, f"{base_name}_rotated.png"),
        'no_shadow': os.path.join(output_folder, f"{base_name}_no_shadow.png"),
        'enhanced': os.path.join(output_folder, f"{base_name}_enhanced.png"),
        'binary': os.path.join(output_folder, f"{base_name}_binary.png")
    }
    
    cv2.imwrite(output_paths['rotated'], corrected_img)
    cv2.imwrite(output_paths['no_shadow'], no_shadow)
    cv2.imwrite(output_paths['enhanced'], enhanced)
    cv2.imwrite(output_paths['binary'], binary)
    
    print(f"  ✓ Preprocessed images saved to: {output_folder}/")
    
    return output_paths

def extract_with_tesseract(file_path, is_preprocessed: bool = False) -> str:
    """
    Method 2/3/4: Extract text using Tesseract OCR with multiple configurations
    
    Args:
        file_path: Path to image or dict of preprocessed images
        is_preprocessed: If True, expects file_path to be a dict of preprocessed images
    """
    if is_preprocessed:
        print("\n[Method 4/4] Using Tesseract OCR on preprocessed images...")
    else:
        print("\n[Method 2/4] Using Tesseract OCR (fallback)...")
    
    # Handle preprocessed images dict
    if isinstance(file_path, dict):
        print("  - Trying preprocessed images...")
        texts = []
        
        for img_type, img_path in file_path.items():
            if os.path.exists(img_path):
                print(f"  - OCR on {img_type}...")
                img = Image.open(img_path)
                texts.append(pytesseract.image_to_string(img, lang='eng+nld', config='--psm 6'))
                texts.append(pytesseract.image_to_string(img, lang='eng+nld', config='--psm 11'))
        
        combined = "\n".join(texts)
        print(f"✓ Tesseract OCR (preprocessed) extracted {len(combined)} characters")
        return combined
    
    # Original Tesseract logic for non-preprocessed images
    img = cv2.imread(file_path)
    
    if img is None:
        print(f"✗ Error: Could not read image file: {file_path}")
        return ""
    
    texts = []
    
    try:
        # 1. Original image with multiple PSM modes
        print("  - Trying original image...")
        original = Image.open(file_path)
        texts.append(pytesseract.image_to_string(original, lang='eng+nld', config='--psm 6'))
        texts.append(pytesseract.image_to_string(original, lang='eng+nld', config='--psm 11'))
        
        # 2. Grayscale
        print("  - Trying grayscale...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        texts.append(pytesseract.image_to_string(Image.fromarray(gray), lang='eng+nld'))
        
        # 3. Binary threshold (Otsu)
        print("  - Trying binary threshold...")
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        texts.append(pytesseract.image_to_string(Image.fromarray(binary), lang='eng+nld'))
        
        # 4. Adaptive threshold
        print("  - Trying adaptive threshold...")
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        texts.append(pytesseract.image_to_string(Image.fromarray(adaptive), lang='eng+nld'))
        
        # 5. Focus on bottom region
        print("  - Trying bottom region...")
        height, width = img.shape[:2]
        bottom_region = img[int(height*0.65):, :]
        texts.append(pytesseract.image_to_string(bottom_region, lang='eng', config='--psm 6'))
        texts.append(pytesseract.image_to_string(bottom_region, lang='eng', config='--psm 11'))
        
        # 6. Top region
        print("  - Trying top region...")
        top_region = img[:int(height*0.7), :]
        texts.append(pytesseract.image_to_string(top_region, lang='eng+nld', config='--psm 6'))
        
        combined = "\n".join(texts)
        print(f"✓ Tesseract OCR extracted {len(combined)} characters")
        return combined
        
    except Exception as e:
        print(f"✗ Tesseract OCR error: {type(e).__name__}: {e}")
        return "\n".join(texts)

def extract_diploma_number(text: str) -> Optional[str]:
    """Extract diploma number from text using progressive pattern matching"""
    if not text or len(text.strip()) < 10:
        return None
    
    diploma_patterns = [
        # Exact B-VCA formats
        (r'\b(\d{7})\s*\.\s*(\d{4})\s+(\d{3,4})\b', 'split', 'Split format: 1234567.1506 294'),
        (r'\b(\d{7})\s*\.\s*(\d{8})\b', 'dot', 'B-VCA format: 1234567.12345678'),
        
        # Pontifex/Other certificates (more flexible)
        (r'(?:registration number|registration|nummer)\s*[:|\s]*(\d{6,7})\s*\.\s*(\d{7,9})', 'dot', 'Registration number'),
        (r'\b(\d{6})\s*\.\s*(\d{8})\b', 'dot', 'Pontifex format: 551715.70727514'),
        (r'\b(\d{6,7})\s*\.\s*(\d{7,9})\b', 'dot', 'Flexible with dot'),
        
        # Other formats
        (r'(\d{7,8})\s*[.,]\s*(\d{7,9})', 'dot', 'With comma or dot separator'),
        # (r'\b(\d{13,16})\b', 'plain', 'Long number without separator'),
    ]
    
    for pattern, format_type, description in diploma_patterns:
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(matches[0], tuple):
                if format_type == 'split' and len(matches[0]) == 3:
                    diploma_num = f"{matches[0][0]}.{matches[0][1]}{matches[0][2]}"
                elif format_type == 'dot' and len(matches[0]) == 2:
                    diploma_num = f"{matches[0][0]}.{matches[0][1]}"
                else:
                    diploma_num = ''.join(str(x) for x in matches[0])
            else:
                diploma_num = matches[0]
            
            diploma_num = diploma_num.replace(' ', '')
            print(f"  ✓ Diploma number found: {diploma_num} ({description})")
            return diploma_num
    
    return None

def extract_diploma_info(text: str) -> Dict[str, str]:
    """Extract all diploma information from text"""
    info = {}
    
    diploma_num = extract_diploma_number(text)
    if diploma_num:
        info['diploma_number'] = diploma_num
    
    name_match = re.search(r'(?:Naam|Name)\s*[:\-]?\s*([A-Z][\.,\s]*[A-Za-z]+(?:\s+[A-Za-z]+)*)', 
                          text, re.IGNORECASE)
    if name_match:
        info['name'] = name_match.group(1).strip()
        print(f"  ✓ Name: {info['name']}")
    
    dob_match = re.search(r'(?:Geboortedatum|Date of birth)\s*[:\-.\s]*(\d{2}[-/]\d{2}[-/]\d{4})', 
                         text, re.IGNORECASE)
    if dob_match:
        info['date_of_birth'] = dob_match.group(1)
        print(f"  ✓ Date of Birth: {info['date_of_birth']}")
    
    issue_match = re.search(r'(?:Afgiftedatum|Date of issuance)\s*[:\-.\s~]*(\d{2}[-/]\d{2}[-/]\d{4})', 
                           text, re.IGNORECASE)
    if issue_match:
        info['date_of_issuance'] = issue_match.group(1)
        print(f"  ✓ Date of Issuance: {info['date_of_issuance']}")
    
    place_match = re.search(r'(?:Plaats van afgifte|Place of issuance)\s*[:\-]?\s*([A-Za-z]+)', 
                           text, re.IGNORECASE)
    if place_match:
        info['place_of_issuance'] = place_match.group(1)
        print(f"  ✓ Place: {info['place_of_issuance']}")
    
    return info

def process_diploma(file_path: str, show_text: bool = False, output_folder: str = "output") -> Dict[str, str]:
    """
    Main function: Process diploma with 4-step fallback strategy + intelligent rotation detection
    
    Strategy:
    1. Try Unstructured API first (fast, works on PDF/images)
    2. If fails, use Tesseract OCR on original file
    3. If still fails AND it's a PDF: Convert PDF to images
    4. Apply intelligent orientation detection (0°, 90°, 180°, 270°)
    5. Apply advanced preprocessing (shadow removal, fine rotation) + Tesseract OCR
    
    Args:
        file_path: Path to diploma image/PDF
        show_text: Whether to print extracted text (for debugging)
        output_folder: Folder to save preprocessed images
    
    Returns:
        Dictionary with extracted diploma information
    """
    if not os.path.exists(file_path):
        print(f"✗ Error: File not found: {file_path}")
        return {}
    
    print("=" * 70)
    print(f"Processing: {os.path.basename(file_path)}")
    print(f"File type: {'PDF' if is_pdf(file_path) else 'Image'}")
    print("=" * 70)
    
    extracted_text = ""
    method_used = "None"
    pdf_images = []  # Store converted PDF images for cleanup
    
    # Step 1: Try Unstructured API
    unstructured_text = extract_with_unstructured(file_path)
    
    if unstructured_text:
        temp_info = extract_diploma_info(unstructured_text)
        
        if 'diploma_number' in temp_info:
            print("✓ Step 1 successful: Unstructured API found diploma number")
            extracted_text = unstructured_text
            method_used = "Step 1: Unstructured API"
            temp_info['extraction_method'] = method_used
            temp_info['file_name'] = os.path.basename(file_path)
            return temp_info
    
    # Step 2: Try Tesseract on original file (if it's an image)
    if not is_pdf(file_path):
        print("⚠ Step 1 failed, trying Step 2 (Tesseract on original image)...")
        tesseract_text = extract_with_tesseract(file_path, is_preprocessed=False)
        
        combined_text = f"{unstructured_text or ''}\n{tesseract_text}"
        temp_info2 = extract_diploma_info(combined_text)
        
        if 'diploma_number' in temp_info2:
            print("✓ Step 2 successful: Tesseract found diploma number")
            extracted_text = combined_text
            method_used = "Step 2: Tesseract OCR"
            temp_info2['extraction_method'] = method_used
            temp_info2['file_name'] = os.path.basename(file_path)
            return temp_info2
        
        # Step 3: Advanced preprocessing on original image (with intelligent rotation)
        print("⚠ Step 2 failed, trying Step 3 (Image Preprocessing with Rotation Detection)...")
        preprocessed_paths = preprocess_image_advanced(file_path, output_folder)
        preprocessed_text = extract_with_tesseract(preprocessed_paths, is_preprocessed=True)
        
        extracted_text = f"{combined_text}\n{preprocessed_text}"
        method_used = "Step 3: Image Preprocessing + Tesseract"
        
    else:
        # It's a PDF - convert to images first
        print("⚠ Step 1 failed on PDF, trying Step 3 (PDF to Image conversion)...")
        print("\n[Method 3/4] Converting PDF to images...")
        
        temp_pdf_folder = os.path.join(output_folder, "temp_pdf_images")
        pdf_images = pdf_to_images(file_path, temp_pdf_folder, dpi=300)
        
        if not pdf_images:
            print("✗ PDF conversion failed")
            return {'extraction_method': 'FAILED - PDF conversion', 'file_name': os.path.basename(file_path)}
        
        # Try Tesseract on converted images WITH ROTATION DETECTION
        all_texts = [unstructured_text or '']
        
        for img_path in pdf_images:
            print(f"\n  Processing converted page: {os.path.basename(img_path)}")
            
            # Step 3a: Try regular Tesseract first
            img_text = extract_with_tesseract(img_path, is_preprocessed=False)
            all_texts.append(img_text)
            
            # Check if we found the diploma number
            temp_check = extract_diploma_info("\n".join(all_texts))
            if 'diploma_number' in temp_check:
                print("✓ Step 3 successful: Found diploma number in converted PDF page")
                extracted_text = "\n".join(all_texts)
                method_used = "Step 3: PDF to Image + Tesseract"
                temp_check['extraction_method'] = method_used
                temp_check['file_name'] = os.path.basename(file_path)
                return temp_check
        
        # Step 4: Advanced preprocessing with INTELLIGENT ROTATION on converted images
        print("\n⚠ Step 3 failed, trying Step 4 (Intelligent Rotation + Preprocessing)...")
        
        for img_path in pdf_images:
            print(f"\n  Preprocessing page: {os.path.basename(img_path)}")
            preprocessed_paths = preprocess_image_advanced(img_path, output_folder)
            preprocessed_text = extract_with_tesseract(preprocessed_paths, is_preprocessed=True)
            all_texts.append(preprocessed_text)
            
            # Check after each page processing
            temp_check = extract_diploma_info("\n".join(all_texts))
            if 'diploma_number' in temp_check:
                print("✓ Step 4 successful: Found diploma number after preprocessing!")
                extracted_text = "\n".join(all_texts)
                method_used = "Step 4: PDF to Image + Rotation Detection + Preprocessing"
                temp_check['extraction_method'] = method_used
                temp_check['file_name'] = os.path.basename(file_path)
                return temp_check
        
        extracted_text = "\n".join(all_texts)
        method_used = "Step 4: PDF to Image + Preprocessing + Tesseract (Failed)"
    
    # Show extracted text if requested
    if show_text:
        print("\n" + "=" * 70)
        print("EXTRACTED TEXT")
        print("=" * 70)
        print(extracted_text[:1000])
        if len(extracted_text) > 1000:
            print(f"... ({len(extracted_text) - 1000} more characters)")
        print("=" * 70)
    
    # Extract final information
    print(f"\nExtracting information using: {method_used}")
    info = extract_diploma_info(extracted_text)
    
    # Add metadata
    info['extraction_method'] = method_used
    info['file_name'] = os.path.basename(file_path)
    
    return info

def main():
    """Process all diploma files from input folder with 4-step strategy"""
    input_folder = "input_data"
    output_folder = "output"
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all files
    test_files = []
    if os.path.exists(input_folder):
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf', '.bmp', '.tiff')):
                    test_files.append(os.path.join(root, file))
    
    print("=" * 70)
    print("ROBUST DUTCH B-VCA DIPLOMA EXTRACTOR")
    print("With Intelligent Rotation Detection (0°, 90°, 180°, 270°)")
    print("=" * 70)
    print("Step 1: Unstructured API (fast, works on PDF/images)")
    print("Step 2: Tesseract OCR on original (images only)")
    print("Step 3: PDF to Image conversion + Tesseract")
    print("Step 4: Intelligent Rotation + Preprocessing + Tesseract")
    print("=" * 70)
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Total files found: {len(test_files)}")
    print("=" * 70)
    
    results = []
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"\n⚠ Skipping {file_path} (not found)")
            continue
        
        try:
            info = process_diploma(file_path, show_text=False, output_folder=output_folder)
            results.append(info)
            
            # Print summary
            print("\n" + "-" * 70)
            print("RESULT SUMMARY")
            print("-" * 70)
            for key, value in info.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
            
            if 'diploma_number' in info:
                print(f"\n✓✓✓ SUCCESS: {info['diploma_number']}")
            else:
                print("\n✗✗✗ FAILED: Could not extract diploma number")
            
            print("-" * 70)
            
        except Exception as e:
            print(f"\n✗ Error processing {file_path}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total files processed: {len(results)}")
    successful = sum(1 for r in results if 'diploma_number' in r)
    print(f"Successfully extracted: {successful}/{len(results)}")
    print("=" * 70)
    
    # Print extraction methods
    print("\nExtraction Methods:")
    print(f"{'File Name':<35s} {'Method':<45s} {'Diploma Number':<20s}")
    print("-" * 100)
    for result in results:
        method = result.get('extraction_method', 'Unknown')
        file_name = result.get('file_name', 'Unknown')[:33]
        diploma = result.get('diploma_number', 'NOT FOUND')[:18]
        print(f"{file_name:<35s} {method:<45s} {diploma:<20s}")
    
    # Count by step
    step1 = sum(1 for r in results if 'Step 1' in r.get('extraction_method', ''))
    step2 = sum(1 for r in results if 'Step 2' in r.get('extraction_method', ''))
    step3 = sum(1 for r in results if 'Step 3' in r.get('extraction_method', ''))
    step4 = sum(1 for r in results if 'Step 4' in r.get('extraction_method', ''))
    
    print("\n" + "=" * 70)
    print("STEP USAGE STATISTICS")
    print("=" * 70)
    print(f"Step 1 (Unstructured API):                    {step1}/{len(results)}")
    print(f"Step 2 (Tesseract on original):               {step2}/{len(results)}")
    print(f"Step 3 (PDF to Image + Tesseract):            {step3}/{len(results)}")
    print(f"Step 4 (+ Rotation Detection + Preprocessing): {step4}/{len(results)}")
    print("=" * 70)

if __name__ == "__main__":
    main() 