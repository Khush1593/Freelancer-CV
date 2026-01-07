import os
import base64
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import fitz  # PyMuPDF

load_dotenv()


class CertificateData(BaseModel):
    """Pydantic model for structured certificate data"""
    name: str = Field(..., description="Full name of the certificate holder")
    DOB: str = Field(..., description="Date of birth in DD/MM/YYYY format")
    date_of_certificate_issue: str = Field(..., description="Certificate issue date in DD/MM/YYYY format")
    validation_upto_date: str = Field(..., description="Certificate expiry date in DD/MM/YYYY format")
    diploma_registration_number: str = Field(..., description="Diploma or registration number")
    
    @field_validator('DOB', 'date_of_certificate_issue', 'validation_upto_date')
    def validate_date_format(cls, v):
        """Validate date format is DD/MM/YYYY or allow 'Not found'"""
        # Allow "Not found" for missing dates
        if v.lower() == "not found":
            return v
        
        if not re.match(r'^\d{2}/\d{2}/\d{4}$', v):
            # Try to parse common formats and convert
            # Handle formats like: DD-MM-YYYY, YYYY-MM-DD, etc.
            date_patterns = [
                (r'(\d{2})-(\d{2})-(\d{4})', r'\1/\2/\3'),  # DD-MM-YYYY
                (r'(\d{4})-(\d{2})-(\d{2})', r'\3/\2/\1'),  # YYYY-MM-DD
                (r'(\d{2})\.(\d{2})\.(\d{4})', r'\1/\2/\3'),  # DD.MM.YYYY
            ]
            
            for pattern, replacement in date_patterns:
                match = re.match(pattern, v)
                if match:
                    v = re.sub(pattern, replacement, v)
                    break
            
            # Final check
            if not re.match(r'^\d{2}/\d{2}/\d{4}$', v):
                raise ValueError(f'Date must be in DD/MM/YYYY format, got: {v}')
        
        return v


def encode_image_to_base64(image_path):
    """Encode local image to base64 data URL"""
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Detect image format
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.png']:
        mime_type = 'image/png'
    else:
        mime_type = 'image/jpeg'
    
    return f"data:{mime_type};base64,{encoded}"


def pdf_to_images_temp(pdf_path, output_folder="temp_vlm_pdf"):
    """
    Convert PDF to images temporarily for VLM processing
    Returns list of image paths
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []
    
    try:
        print(f"\n📄 Converting PDF to images: {os.path.basename(pdf_path)}")
        
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        print(f"   Pages: {total_pages}")
        
        # Convert each page to image
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            
            # Render at high DPI for better quality
            zoom = 300 / 72  # 300 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Save as PNG
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            image_path = os.path.join(output_folder, f'{base_name}_page_{page_num + 1}.png')
            
            pix.save(image_path)
            image_paths.append(image_path)
            
            print(f"   ✓ Page {page_num + 1}/{total_pages}: {os.path.basename(image_path)}")
        
        pdf_document.close()
        print(f"✅ Converted {total_pages} pages!")
        return image_paths
        
    except Exception as e:
        print(f"❌ Error converting PDF: {e}")
        return []


def extract_certificate_data_vlm(image_path):
    """
    Extract structured certificate data using VLM with Pydantic validation
    Returns CertificateData object or None
    """
    
    # Initialize client
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=os.environ["HF_TOKEN"],
    )
    
    # Encode image
    print(f"\n📷 Encoding image: {os.path.basename(image_path)}")
    image_data_url = encode_image_to_base64(image_path)
    print(f"✓ Image encoded (size: {len(image_data_url)} chars)")
    
    print("\n🤖 Sending request to HuggingFace (Qwen3-VL)...")
    
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct:novita",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Extract ALL information from this Dutch safety certificate/diploma and return ONLY a JSON object with this EXACT structure:

{
  "name": "Full name of person",
  "DOB": "Date of birth in DD/MM/YYYY format",
  "date_of_certificate_issue": "Issue date in DD/MM/YYYY format",
  "validation_upto_date": "Expiry/valid until date in DD/MM/YYYY format",
  "diploma_registration_number": "Registration or diploma number (format: XXXXXX.XXXXXXXX)"
}

IMPORTANT:
- ALL dates MUST be in DD/MM/YYYY format (e.g., 14/04/1997)
- If a field is not found, use "Not found" as the value
- Look for fields like: "Naam", "Name", "Geboortedatum", "Date of Birth", "Geldig tot", "Valid until", "Nummer", "Registration number"
- Return ONLY the JSON object, no other text"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=512,
            temperature=0.1
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        print("\n" + "="*70)
        print("VLM RAW RESPONSE")
        print("="*70)
        print(response_text)
        print("="*70)
        
        # Extract JSON from response (in case there's extra text)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        else:
            json_text = response_text
        
        # Parse JSON
        try:
            data_dict = json.loads(json_text)
            
            # Validate with Pydantic
            certificate_data = CertificateData(**data_dict)
            
            print("\n" + "="*70)
            print("✅ VALIDATED CERTIFICATE DATA")
            print("="*70)
            print(json.dumps(certificate_data.model_dump(), indent=2))
            print("="*70)
            
            return certificate_data
            
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON Parse Error: {e}")
            print(f"Attempted to parse: {json_text[:200]}")
            return None
            
        except Exception as e:
            print(f"\n❌ Pydantic Validation Error: {e}")
            print(f"Data received: {data_dict}")
            return None
        
    except Exception as e:
        print(f"\n❌ API Error: {type(e).__name__}: {e}")
        return None


def process_file_with_vlm(file_path):
    """
    Process either PDF or Image file with VLM
    Returns structured CertificateData or None
    """
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print("="*70)
    print(f"PROCESSING: {os.path.basename(file_path)}")
    print("="*70)
    
    # Check file type
    is_pdf = file_path.lower().endswith('.pdf')
    
    if is_pdf:
        # Convert PDF to images first
        print("📄 File type: PDF")
        image_paths = pdf_to_images_temp(file_path)
        
        if not image_paths:
            print("❌ PDF conversion failed")
            return None
        
        # Process first page (usually contains all info)
        # You can modify to process all pages if needed
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(image_paths[0])}")
        print("="*70)
        
        certificate_data = extract_certificate_data_vlm(image_paths[0])
        
        return certificate_data
            
    else:
        # Direct image processing
        print("🖼️  File type: Image")
        certificate_data = extract_certificate_data_vlm(file_path)
        
        return certificate_data


if __name__ == "__main__":
    # Test files
    test_files = [
        r"input_data\VCASwiderek.jpg"
    ]
    
    # Choose file to test
    test_file = test_files[0]  # Change index to test different files
    
        # Moved to src/VLM_image_call.py