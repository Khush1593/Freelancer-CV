"""
Insurance Document Data Extraction using VLM
Handles insurance documents (PDF/Image) including long multi-page PDFs
"""

import os
import base64
import json
import re
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import fitz  # PyMuPDF
from PIL import Image

load_dotenv()


def _log(debug: list | None, message: str):
    """Print to stdout and optionally collect debug messages into a list."""
    import sys
    try:
        # Always print to console for live debugging
        print(message, flush=True)
        sys.stdout.flush()
    finally:
        if debug is not None:
            debug.append(message)


def _get_hf_api_key(debug: list | None = None) -> str | None:
    """Return an API key for the Hugging Face Inference Router.
    Tries common env vars in order: HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, OPENAI_API_KEY.
    Logs which one is used. Returns None if not found.
    """
    candidates = [
        ("HF_TOKEN", os.environ.get("HF_TOKEN")),
        ("HUGGINGFACEHUB_API_TOKEN", os.environ.get("HUGGINGFACEHUB_API_TOKEN")),
        ("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY")),
    ]
    present = [name for name, value in candidates if value]
    if present:
        _log(debug, f"🔎 Detected API key env present: {present}")
    else:
        _log(debug, "🔎 No API key env vars detected among HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, OPENAI_API_KEY")
    for name, value in candidates:
        if value:
            _log(debug, f"🔐 Using API key from {name}")
            return value
    _log(debug, "❌ No API key found. Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN (preferred), or OPENAI_API_KEY.")
    return None


class Coverage(BaseModel):
    """Model for individual coverage within an insurance policy"""
    coverage_type: str = Field(..., description="Type of coverage (e.g., liability, property)")
    policy_number: str = Field(default="Not found", description="Policy number for this coverage")
    amount: str = Field(default="Not found", description="Coverage amount")


class InsuranceData(BaseModel):
    """Pydantic model for structured insurance document data"""
    # Primary fields (allow "Not found" as valid value)
    freelancer_name: str = Field(default="Not found", description="Name of the freelancer/insured person")
    insurance_contract_number: str = Field(default="Not found", description="Insurance or contract number")
    start_date: str = Field(default="Not found", description="Start date of insurance in DD/MM/YYYY format")
    end_date: str = Field(default="Not found", description="End date of insurance in DD/MM/YYYY format")
    
    # Coverage information
    coverages: List[Coverage] = Field(default_factory=list, description="List of coverages with amounts")
    
    # Optional fields for data collection
    insurance_company_name: str = Field(default="Not found", description="Name of insurance company")
    insurance_name: str = Field(default="Not found", description="Name/type of the insurance policy")
    insurance_premium: str = Field(default="Not found", description="Insurance premium amount per period")
    conditions_summary: str = Field(default="Not found", description="Brief summary of insurance conditions")
    
    # Deprecated fields (kept for backward compatibility)
    max_insured_per_event: str = Field(default="Not found", description="[DEPRECATED] Use coverages instead")
    max_insured_per_year: str = Field(default="Not found", description="[DEPRECATED] Use coverages instead")
    
    @field_validator('start_date', 'end_date')
    def validate_date_format(cls, v):
        """Validate date format is DD/MM/YYYY or allow 'Not found'"""
        if v.lower() == "not found":
            return v
        
        if not re.match(r'^\d{2}/\d{2}/\d{4}$', v):
            # Try to parse common formats and convert
            date_patterns = [
                (r'(\d{2})-(\d{2})-(\d{4})', r'\1/\2/\3'),  # DD-MM-YYYY
                (r'(\d{4})-(\d{2})-(\d{2})', r'\3/\2/\1'),  # YYYY-MM-DD
                (r'(\d{2})\.(\d{2})\.(\d{4})', r'\1/\2/\3'),  # DD.MM.YYYY
            ]
            
            for pattern, replacement in date_patterns:
                match = re.match(pattern, v)
                if match:
                    print(f"[validator] Converting date '{v}' using pattern '{pattern}'")
                    v = re.sub(pattern, replacement, v)
                    break
            
            # Final check
            if not re.match(r'^\d{2}/\d{2}/\d{4}$', v):
                raise ValueError(f'Date must be in DD/MM/YYYY format, got: {v}')
        
        return v


def compress_image_if_needed(image_path, max_size_kb=200):
    """Compress image if it's too large to avoid 413 errors"""
    file_size_kb = os.path.getsize(image_path) / 1024
    try:
        with Image.open(image_path) as _probe:
            print(f"   🖼️  Original image dims: {_probe.width}x{_probe.height}")
    except Exception:
        pass
    
    if file_size_kb <= max_size_kb:
        return image_path  # No compression needed
    
    print(f"   ⚠️  Image is {file_size_kb:.1f} KB, compressing...")
    
    # Open image
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize if too large
    max_width = 1024
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.LANCZOS)
        print(f"   📏 Resized image to {max_width}x{new_height}")
    
    # Create compressed version
    compressed_path = image_path.rsplit('.', 1)[0] + '_compressed.jpg'
    
    # Try different quality levels until under max_size_kb
    # Start with higher quality (85) and minimum quality of 60 (not 20)
    quality = 85
    while quality >= 60:
        img.save(compressed_path, "JPEG", quality=quality, optimize=True)
        file_size_kb = os.path.getsize(compressed_path) / 1024
        if file_size_kb <= max_size_kb:
            break
        quality -= 10
    
    print(f"   ✓ Compressed to {file_size_kb:.1f} KB (quality={quality})")
    return compressed_path


def encode_image_to_base64(image_path):
    """Encode local image to base64 data URL with compression if needed"""
    # Compress if needed - allow up to 500KB per image (increased from 200KB)
    compressed_path = compress_image_if_needed(image_path, max_size_kb=500)
    try:
        orig_kb = os.path.getsize(image_path) / 1024
        comp_kb = os.path.getsize(compressed_path) / 1024
        print(f"   🗜️  Encoding image (orig: {orig_kb:.1f} KB, used: {comp_kb:.1f} KB): {os.path.basename(compressed_path)}")
    except Exception:
        pass
    
    with open(compressed_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Always use JPEG mime type since we compress to JPEG
    return f"data:image/jpeg;base64,{encoded}"


def pdf_to_images_temp(pdf_path, output_folder="temp_insurance_pdf", max_pages=50, debug: list | None = None):
    """
    Convert PDF to images temporarily for VLM processing
    Handles long PDFs (up to max_pages)
    Returns list of image paths
    """
    os.makedirs(output_folder, exist_ok=True)
    image_paths = []
    
    try:
        _log(debug, f"\n📄 Converting PDF to images: {os.path.basename(pdf_path)}")
        
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        try:
            size_bytes = os.path.getsize(pdf_path)
            _log(debug, f"   📦 PDF size: {size_bytes/1024:.1f} KB")
        except Exception:
            pass
        
        # Limit pages if needed
        pages_to_process = min(total_pages, max_pages)
        
        if total_pages > max_pages:
            _log(debug, f"   ⚠️  PDF has {total_pages} pages, processing first {max_pages} pages")
        else:
            _log(debug, f"   Pages: {total_pages}")
        
        # Convert each page to image
        for page_num in range(pages_to_process):
            page = pdf_document[page_num]
            
            # Render at higher DPI for better text readability
            # 150 DPI is a good balance - text is clear, file size reasonable
            zoom = 150 / 72  # 150 DPI (good for document reading)
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            _log(debug, f"   🧮 Rendered page {page_num+1} at 150 DPI -> {pix.width}x{pix.height}")
            
            # Save as JPEG with compression for smaller file size
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            image_path = os.path.join(output_folder, f'{base_name}_page_{page_num + 1}.jpg')
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Resize if very large (max width 2048px instead of 1024px for better quality)
            max_width = 2048
            if img.width > max_width:
                ratio = max_width / img.width
                new_height = int(img.height * ratio)
                img = img.resize((max_width, new_height), Image.LANCZOS)
                _log(debug, f"   📏 Resized to {max_width}x{new_height}")
            
            # Save with good quality (85 is high quality, readable text)
            img.save(image_path, "JPEG", quality=85, optimize=True)
            
            # Check file size - we'll allow up to 500KB per page for better quality
            file_size_kb = os.path.getsize(image_path) / 1024
            _log(debug, f"   💾 Saved page {page_num+1}: {file_size_kb:.1f} KB")
            
            # Only compress further if exceeds 500KB (not as aggressive as before)
            if file_size_kb > 500:
                quality = 75
                while file_size_kb > 500 and quality > 50:
                    img.save(image_path, "JPEG", quality=quality, optimize=True)
                    file_size_kb = os.path.getsize(image_path) / 1024
                    quality -= 10
                _log(debug, f"   🗜️ Compressed to {file_size_kb:.1f} KB (quality={quality+10})")
            
            image_paths.append(image_path)
            _log(debug, f"   ✓ Page {page_num + 1}/{pages_to_process}: {os.path.basename(image_path)} ({file_size_kb:.1f} KB)")
        
        pdf_document.close()
        _log(debug, f"✅ Converted {pages_to_process} pages!")
        return image_paths
        
    except Exception as e:
        _log(debug, f"❌ Error converting PDF: {e}")
        return []


def extract_page_content_lossless(image_path, page_num: int, debug: list | None = None) -> Optional[str]:
    """
    Step 1: Extract all text/content from a page image in a lossless manner.
    Returns the raw extracted content as a string.
    """
    try:
        file_size_kb = os.path.getsize(image_path) / 1024
        _log(debug, f"   📄 [Step 1] Extracting content from page {page_num}: {os.path.basename(image_path)} ({file_size_kb:.1f} KB)")
    except Exception:
        _log(debug, f"   ⚠️ Could not stat image file: {image_path}")
    
    # Initialize client
    api_key = _get_hf_api_key(debug)
    if not api_key:
        return None
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        _log(debug, "   🌐 OpenAI client initialized (HF Router)")
    except Exception as e:
        _log(debug, f"   ❌ Failed to initialize OpenAI client: {e}")
        return None
    
    # Encode image
    _log(debug, f"\n📷 [Step 1] Processing page {page_num}: {os.path.basename(image_path)}")
    image_data_url = encode_image_to_base64(image_path)
    _log(debug, f"   🔗 Encoded image length: {len(image_data_url)} chars")
    
    try:
        model_name = "Qwen/Qwen3-VL-8B-Instruct:novita"
        _log(debug, f"   🤖 Calling model: {model_name} (content extraction)")
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are an INSURANCE DOCUMENT VISION–LANGUAGE EXTRACTION ENGINE.

You analyze ONE SINGLE PAGE of an insurance document using:
- visual layout (logos, headers, footers, emphasis, positioning)
- visible textual content

Your task is to produce a SEMANTIC PAGE TRANSCRIPTION that preserves
ALL visible information while adding LIMITED, EVIDENCE-BASED insurance context.

PAGE CLASSIFICATION (MANDATORY):
Classify the page as ONE of:
- POLICY_OVERVIEW
- COVERAGE_DETAILS
- PREMIUM_CALCULATION
- CONDITIONS
- OTHER

OUTPUT FORMAT:
Start with:
=== TYPE: <PAGE_TYPE> ===

SECTIONED SEMANTIC EXTRACTION (MANDATORY):

Use the following sections when applicable, Do NOT invent sections:

[HEADER / LETTERHEAD]
- List ALL visible logos, brand names, and header text.
- For EACH logo or brand:
- Name (exactly as shown)
- Role (choose ONE):
  • Brand / Label
  • Intermediary / Broker
  • Authorized agent (gevolmachtigde)
  • Role unclear
- Evidence:
  • “Explicitly stated” OR
  • “Inferred from layout only”

IMPORTANT:
- Logos, branding, addresses, or layout position alone
  MUST NOT be used to identify the risk carrier / insurer.
- Do NOT assign “Risk carrier / Insurer” in this section.

[DOCUMENT IDENTIFICATION]
- Document title
- Document language
- Page number if visible

[PARTIES]
List parties ONLY when visible on the page.

For each party include:
- Name
- Role (exact wording from document when available)

Possible roles include (not exhaustive):
- Policyholder / Insured (verzekeringnemer)
- Risk carrier / Insurer
- Intermediary / Broker
- Authorized agent (gevolmachtigde)
- Beneficiary
- Other explicitly stated role

STRICT ROLE RULES:
- Assign “Risk carrier / Insurer” ONLY if explicitly stated in legal text
  (e.g., “verzekeraar”, “als gevolmachtigde van”).
- Footer legal statements override header branding.
- Do NOT infer legal roles from logos.
- Do NOT contradict yourself.

[POLICY INFORMATION]
- Insurance type
- Policy number
- Policy conditions
- Contract duration
- Start date
- Modification date and reason

[COVERAGE & LIMITS]
- Insured profession / activity
- Coverage limits
- Annual maximums
- Deductibles (eigen risico)

Preserve monetary formatting exactly.

[PREMIUM INFORMATION]
- Base premium
- Payment frequency
- Taxes or exclusions (if visible)

[CLAUSES & NOTES]
- Clauses
- Remarks
- Footnotes
- Special conditions

[FOOTER]
- Location
- Date
- Signatory statements
- Capacity percentages
- Explicit statements identifying the risk carrier

RULES:
- Extract ALL visible content from THIS PAGE ONLY.
- Preserve original wording and language.
- Preserve labels with values.
- Do NOT summarize.
- Do NOT merge with other pages.
- Do NOT invent or assume missing data.
- LIMITED interpretation is allowed ONLY to:
  • identify roles of entities
  • group content into insurance-relevant sections
- When uncertain, state uncertainty explicitly.
- If present information about the end_date then must include.

Only output the extracted semantic transcription.
"""
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
            max_tokens=1024,
            temperature=0.1
        )
        
        content = completion.choices[0].message.content.strip()
        _log(debug, f"   ✓ Extracted content length: {len(content)} chars")
        _log(debug, f"   📝 Content preview: {content}...")
        print("____________Content completed_____________________")
        return content
        
    except Exception as e:
        _log(debug, f"❌ Error extracting content from page {page_num}: {e} ({type(e).__name__})")
        import traceback
        _log(debug, f"Full error: {traceback.format_exc()}")
        return None

def generate_final_json_from_content(all_content: str, debug: list | None = None) -> Optional[dict]:
    """
    Step 2: Generate final insurance JSON from aggregated page content.
    Returns dict with insurance data or None.
    """
    _log(debug, "\n" + "="*70)
    _log(debug, "[Step 2] GENERATING FINAL JSON FROM AGGREGATED CONTENT")
    _log(debug, "="*70)
    _log(debug, f"   📊 Total content length: {len(all_content)} chars")
    
    # Initialize client
    api_key = _get_hf_api_key(debug)
    if not api_key:
        return None
    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )
        _log(debug, "   🌐 OpenAI client initialized (HF Router)")
    except Exception as e:
        _log(debug, f"   ❌ Failed to initialize OpenAI client: {e}")
        return None
    
    try:
        model_name = "Qwen/Qwen3-VL-8B-Instruct:novita"
        _log(debug, f"   🤖 Calling model: {model_name} (JSON generation)")

        prompt = f"""You are an insurance analysis engine.

You are given extracted insurance-related content from all pages of a policy document.

TASK:
Generate a final insurance JSON using ONLY the provided content.

RULES:
- Do NOT guess or infer missing data, EXCEPT you MAY calculate an end_date from an explicitly stated contract duration and a known start_date.
- Prefer values explicitly marked as including tax.
- Support multiple coverage types as a list in the "coverages" field.
- Each coverage should have: coverage_type, policy_number (if available), and amount.
- If no separate coverages are found, leave "coverages" as an empty array [].
- Dates must be DD/MM/YYYY format.
- For end_date:
    - If an explicit end date appears in the content, use that value.
    - Otherwise, if the document clearly states a duration (for example "for 1 year", "2 years", "6 months") and the start_date is known, compute end_date as start_date plus that duration and return it in DD/MM/YYYY format.
    - If neither an explicit end date nor a clear duration is present, set end_date to "Not found".
- If a field is missing, use "Not found".

OUTPUT FORMAT:
Return ONLY a JSON object with this EXACT structure:
{{
  "freelancer_name": "Name of the freelancer/insured person",
  "insurance_contract_number": "Insurance or contract number",
  "start_date": "Start date in DD/MM/YYYY format",
  "end_date": "End date in DD/MM/YYYY format",
  "coverages": [
    {{
      "coverage_type": "Type of coverage (e.g., Liability, Property, Professional Indemnity)",
      "policy_number": "Policy number for this specific coverage or Not found",
      "amount": "Coverage amount with currency"
    }}
  ],
  "insurance_company_name": "Name of insurance company",
  "insurance_name": "Name or type of the insurance policy",
  "insurance_premium": "Premium amount per period (including tax if specified)",
  "conditions_summary": "Brief summary of key insurance conditions",
  "max_insured_per_event": "Not found",
  "max_insured_per_year": "Not found"
}}

NOTE: If there are multiple coverages (e.g., different types like liability, property, etc.), include ALL of them in the "coverages" array. If there's only general insurance info without specific coverages, keep "coverages" as [].

--- EXTRACTED CONTENT FROM ALL PAGES ---
{all_content}
--- END OF CONTENT ---

Generate the JSON now:"""
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1024,
            temperature=0.1
        )
        
        response_text = completion.choices[0].message.content.strip()
        _log(debug, f"   API Response len: {len(response_text)}")
        _log(debug, f"   API Response (preview): {response_text[:300]}...")
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
            try:
                data_dict = json.loads(json_text)
            except Exception as je:
                _log(debug, f"   ❌ JSON decode error: {je}")
                _log(debug, f"   JSON text preview: {json_text[:200]}...")
                return None
            _log(debug, f"   ✓ Generated {len(data_dict)} fields")
            return data_dict
        else:
            _log(debug, f"   ⚠️  No JSON found in response")
        
        return None
        
    except Exception as e:
        _log(debug, f"❌ Error generating final JSON: {e} ({type(e).__name__})")
        import traceback
        _log(debug, f"Full error: {traceback.format_exc()}")
        return None

def extract_insurance_data_vlm(file_path, max_pages=50, debug: list | None = None):
    """
    Extract structured insurance data using VLM with Pydantic validation
    Handles multi-page PDFs
    Returns InsuranceData object or None
    """
    
    if not os.path.exists(file_path):
        _log(debug, f"❌ File not found: {file_path}")
        return None
    
    _log(debug, "="*70)
    _log(debug, f"PROCESSING INSURANCE DOCUMENT: {os.path.basename(file_path)}")
    _log(debug, "="*70)
    
    # Check file type
    try:
        size_b = os.path.getsize(file_path)
        _log(debug, f"   📥 Input file size: {size_b/1024:.1f} KB")
    except Exception:
        pass
    is_pdf = file_path.lower().endswith('.pdf')
    
    if is_pdf:
        # Convert PDF to images
        _log(debug, "📄 File type: PDF")
        image_paths = pdf_to_images_temp(file_path, max_pages=max_pages, debug=debug)
        
        if not image_paths:
            _log(debug, "❌ PDF conversion failed")
            return None
        
        # TWO-STEP PROCESS:
        # Step 1: Extract content from all pages
        _log(debug, "\n" + "="*70)
        _log(debug, "STEP 1: EXTRACTING CONTENT FROM ALL PAGES")
        _log(debug, "="*70)
        all_page_contents = []
        for idx, img_path in enumerate(image_paths, start=1):
            content = extract_page_content_lossless(img_path, page_num=idx, debug=debug)
            if content:
                all_page_contents.append(f"\n--- PAGE {idx} ---\n{content}")
                _log(debug, f"   ✅ Page {idx} content extracted ({len(content)} chars)")
            else:
                _log(debug, f"   ⚠️ No content from page {idx}")
        
        if not all_page_contents:
            _log(debug, "❌ No content extracted from any page")
            return None
        
        # Step 2: Aggregate all content and generate final JSON
        aggregated_content = "\n".join(all_page_contents)
        _log(debug, f"\n📦 Aggregated content from {len(all_page_contents)} pages (total: {len(aggregated_content)} chars)")
        
        merged_data = generate_final_json_from_content(aggregated_content, debug=debug)
        
        if not merged_data:
            _log(debug, "❌ Failed to generate JSON from aggregated content")
            return None
        
    else:
        # Direct image processing using 2-step approach
        _log(debug, "🖼️  File type: Image")
        
        # Step 1: Extract content
        _log(debug, "\n" + "="*70)
        _log(debug, "STEP 1: EXTRACTING CONTENT FROM IMAGE")
        _log(debug, "="*70)
        content = extract_page_content_lossless(file_path, page_num=1, debug=debug)
        
        if not content:
            _log(debug, "❌ Failed to extract content from image")
            return None
        
        # Step 2: Generate JSON from content
        merged_data = generate_final_json_from_content(content, debug=debug)
        
        if not merged_data:
            _log(debug, "❌ Failed to generate JSON from image content")
            return None
    
    # Validate with Pydantic
    try:
        _log(debug, f"\n📋 Validating data: {json.dumps(merged_data, indent=2)}")
        insurance_data = InsuranceData(**merged_data)
        _log(debug, "\n" + "="*70)
        _log(debug, "✅ VALIDATED INSURANCE DATA")
        _log(debug, "="*70)
        _log(debug, json.dumps(insurance_data.model_dump(), indent=2))
        _log(debug, "="*70)
        
        return insurance_data
        
    except Exception as e:
        _log(debug, f"\n❌ Pydantic Validation Error: {e}")
        _log(debug, f"Data received: {json.dumps(merged_data, indent=2)}")
        import traceback
        _log(debug, f"Traceback: {traceback.format_exc()}")
        return None


def process_insurance_file_with_debug(file_path, max_pages=50):
    """Wrapper that returns (data, debug)."""
    debug: list[str] = []
    data = extract_insurance_data_vlm(file_path, max_pages=max_pages, debug=debug)
    return data, debug


def process_insurance_file(file_path, max_pages=50):
    """
    Main entry point for insurance document processing
    Returns InsuranceData or None
    """
    return extract_insurance_data_vlm(file_path, max_pages=max_pages)
# if __name__ == "__main__":
#     # Test files
#     test_files = [
#         # r"Insurance_data\Polis_en_voorwaarden_BN180001982aansprakelijkheidbedrijven.pdf",
#         r"Insurance_data\IMG_5720.png"
#     ]
    
#     # Test with first available file
#     for test_file in test_files:
#         if os.path.exists(test_file):
#             print(f"\n{'='*70}")
#             print(f"Testing with: {test_file}")
#             print(f"{'='*70}")
            
#             result = process_insurance_file(test_file, max_pages=50)
            
#             if result:
#                 print("\n" + "="*70)
#                 print("FINAL JSON OUTPUT")
#                 print("="*70)
#                 print(json.dumps(result.model_dump(), indent=2))
#                 print("="*70)
#             else:
#                 print("\n❌ Failed to extract insurance data")
            
#             break
#     else:
#         print("⚠️  No test files found. Please add insurance documents to input_data folder.")
