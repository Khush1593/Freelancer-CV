"""
Comprehensive Dutch B-VCA Diploma Processor
Combines multiple extraction methods to find diploma numbers
"""

import re
import os
from typing import Optional, Dict, List
import pytesseract
from PIL import Image
import cv2
import numpy as np
import unstructured_client
from unstructured_client.models import shared
from unstructured_client.models.operations import PartitionRequest
from dotenv import load_dotenv

load_dotenv()

def extract_with_tesseract(image_path: str) -> str:
    """Extract text using Tesseract OCR with multiple configurations"""
    img = cv2.imread(image_path)
    
    # Try different image preprocessing techniques
    texts = []
    
    # 1. Original image
    texts.append(pytesseract.image_to_string(Image.open(image_path), lang='eng+nld'))
    
    # 2. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texts.append(pytesseract.image_to_string(Image.fromarray(gray), lang='eng+nld'))
    
    # 3. Binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    texts.append(pytesseract.image_to_string(Image.fromarray(binary), lang='eng+nld'))
    
    # 4. Inverted binary
    _, inv_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    texts.append(pytesseract.image_to_string(Image.fromarray(inv_binary), lang='eng+nld'))
    
    # 5. Focus on bottom region where diploma numbers usually are
    height, width = img.shape[:2]
    bottom_region = img[int(height*0.7):, :]  # Bottom 30%
    texts.append(pytesseract.image_to_string(bottom_region, lang='eng', config='--psm 6'))
    
    # 6. Try with digits-only mode on bottom region
    texts.append(pytesseract.image_to_string(bottom_region, lang='eng', config='--psm 6 digits'))
    
    return "\n".join(texts)

def extract_with_unstructured(image_path: str) -> str:
    """Extract text using Unstructured API"""
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        return ""
    
    try:
        client = unstructured_client.UnstructuredClient(api_key_auth=api_key)
        
        with open(image_path, "rb") as f:
            req = PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=f.read(),
                        file_name=os.path.basename(image_path)
                    ),
                    strategy=shared.Strategy.HI_RES,
                    languages=["nld", "eng"],
                    ocr_languages=["nld", "eng"],
                )
            )
            response = client.general.partition(request=req)
            
        if hasattr(response, 'elements') and response.elements:
            return "\n".join([element.get("text", "") for element in response.elements])
    except:
        pass
    
    return ""

def extract_diploma_info(image_path: str) -> Dict[str, any]:
    """
    Extract all diploma information using multiple methods
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return {}
    
    print(f"Processing: {image_path}")
    print("=" * 60)
    
    # Collect text from all sources
    print("\n[1/2] Extracting with Tesseract OCR...")
    tesseract_text = extract_with_tesseract(image_path)
    
    print("[2/2] Extracting with Unstructured API...")
    unstructured_text = extract_with_unstructured(image_path)
    
    # Combine all text
    combined_text = f"{tesseract_text}\n{unstructured_text}"
    
    print("\n" + "=" * 60)
    print("FULL EXTRACTED TEXT")
    print("=" * 60)
    print(combined_text)
    print("=" * 60)
    
    # Extract information with patterns
    info = {}
    
    # Diploma number patterns (prioritized)
    diploma_patterns = [
        (r'\b(\d{7})\s*\.\s*(\d{4})\s+(\d{3,4})\b', 'Split format: 1234567.1234 567'),
        (r'\b(\d{7})\.\d{4}\s+\d{3,4}\b', 'Format with space: 1234567.1234 567'),
        (r'\b(\d{7}\.\d{8})\b', 'Exact format: 1234567.12345678'),
        (r'\b(\d{6,8})\s*\.\s*(\d{7,9})\b', 'With spaces around dot'),
        (r'(\d{7,8})\s*[.,]\s*(\d{7,9})', 'With comma or dot separator'),
        (r'\b(\d{13,16})\b', 'Long number without separator'),
    ]
    
    print("\nSearching for diploma number...")
    for pattern, desc in diploma_patterns:
        matches = re.findall(pattern, combined_text)
        if matches:
            if isinstance(matches[0], tuple):
                # Reconstruct diploma number with dot separator
                if len(matches[0]) == 3:
                    # Format: 1511238.15063294 (7 digits . 4 digits + 3-4 digits)
                    diploma_num = f"{matches[0][0]}.{matches[0][1]}{matches[0][2]}"
                elif len(matches[0]) == 2:
                    diploma_num = f"{matches[0][0]}.{matches[0][1]}"
                else:
                    diploma_num = ''.join(str(x) for x in matches[0])
            else:
                diploma_num = matches[0]
            
            diploma_num = diploma_num.replace(' ', '')
            info['diploma_number'] = diploma_num
            print(f"✓ Found: {diploma_num} (Pattern: {desc})")
            break
    
    # Name
    name_patterns = [
        r'(?:Naam|Name)\s*[:\-]?\s*([A-Z][\.,\s]*[A-Za-z]+(?:\s+[A-Za-z]+)*)',
    ]
    for pattern in name_patterns:
        match = re.search(pattern, combined_text, re.IGNORECASE)
        if match:
            info['name'] = match.group(1).strip()
            print(f"✓ Name: {info['name']}")
            break
    
    # Date of birth
    dob_match = re.search(r'(?:Geboortedatum|Date of birth)\s*[:\-.\s]*(\d{2}[-/]\d{2}[-/]\d{4})', combined_text, re.IGNORECASE)
    if dob_match:
        info['date_of_birth'] = dob_match.group(1)
        print(f"✓ Date of Birth: {info['date_of_birth']}")
    
    # Date of issuance
    issue_match = re.search(r'(?:Afgiftedatum|Date of issuance)\s*[:\-.\s~]*(\d{2}[-/]\d{2}[-/]\d{4})', combined_text, re.IGNORECASE)
    if issue_match:
        info['date_of_issuance'] = issue_match.group(1)
        print(f"✓ Date of Issuance: {info['date_of_issuance']}")
    
    # Place
    place_match = re.search(r'(?:Plaats van afgifte|Place of issuance)\s*[:\-]?\s*([A-Za-z]+)', combined_text, re.IGNORECASE)
    if place_match:
        info['place_of_issuance'] = place_match.group(1)
        print(f"✓ Place: {info['place_of_issuance']}")
    
    # If still no diploma number, show all long numbers found
    if 'diploma_number' not in info:
        print("\n⚠ Diploma number not found. All numbers with 6+ digits:")
        all_numbers = re.findall(r'\b\d{6,}\b', combined_text)
        for num in set(all_numbers):
            print(f"   - {num}")
        
        # Also check for patterns with dots/periods
        numbers_with_dot = re.findall(r'\d+\.\d+', combined_text)
        if numbers_with_dot:
            print("\n  Numbers with decimal points:")
            for num in set(numbers_with_dot):
                print(f"   - {num}")
    
    return info

def main():
    file_to_process = "document.png"
    
    print("=" * 60)
    print("DUTCH B-VCA DIPLOMA PROCESSOR")
    print("=" * 60)
    
    info = extract_diploma_info(file_to_process)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    if info:
        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        if 'diploma_number' in info:
            print(f"\n✓✓✓ SUCCESS: Diploma Number = {info['diploma_number']}")
        else:
            print("\n✗✗✗ Could not extract diploma number")
            print("Please check the 'All numbers' list above")
    else:
        print("✗ No information extracted")

if __name__ == "__main__":
    main()
