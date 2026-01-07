// Moved to archive/old_code/lang_unstr_run.py
"""
Diploma Number Detection using Unstructured Client API

Requirements:
- Virtual environment (venv) with all dependencies installed
- .env file with UNSTRUCTURED_API_KEY set

To run:
    & "C:/Users/kpatel/Desktop/Text detection using EAST/venv/Scripts/python.exe" lang_unstr_run.py

Dependencies in venv:
- unstructured-client
- python-dotenv
- requests and other standard packages
"""

import re
import os
from typing import Optional
# from unstructured_client import unstructured_client
import unstructured_client 
from unstructured_client.models import shared
from unstructured_client.models.operations import PartitionRequest
from dotenv import load_dotenv
load_dotenv()
import mimetypes

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def partition_pdf_api(file_path: str) -> Optional[str]:
    """
    Processes a Dutch B-VCA diploma file using the Unstructured API and extracts the diploma number.
    Expects format like: 1511238.15063294
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    # Check if API key is set
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        print("Error: UNSTRUCTURED_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return None
    
    content_type = get_mime_type(file_path)
    print(f"Processing file: {file_path}")
    print(f"Content type: {content_type}")

    # --- Stage 1: Call Unstructured API for Text Extraction ---
    try:
        # Initialize client with API key
        client = unstructured_client.UnstructuredClient(
            api_key_auth=api_key
        )
        
        with open(file_path, "rb") as f:
            file_content = f.read()
            
            # Create a PartitionRequest with the file
            # Use hi_res strategy for better OCR on images with text
            req = PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=file_content,
                        file_name=os.path.basename(file_path)
                    ),
                    strategy=shared.Strategy.HI_RES,
                    languages=["nld"],  # Dutch language code
                    ocr_languages=["nld"],  # OCR in Dutch
                )
            )
            
            print("Sending request to Unstructured API...")
            response = client.general.partition(request=req)

        # Check if response has elements
        if not hasattr(response, 'elements') or not response.elements:
            print("Warning: No elements returned from API")
            return None
            
        # Combine all elements into a single string for searching
        full_text = "\n".join([element.get("text", "") for element in response.elements])
        print(f"Extracted text length: {len(full_text)} characters")

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except PermissionError:
        print(f"Error: Permission denied when reading file: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during Unstructured API call: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())
        return None

    # --- Stage 2: Regex Matching (Local and fast) ---
    
    if not full_text or full_text.strip() == "":
        print("Warning: Extracted text is empty")
        return None
    
    print("\n--- Extracted Text Preview ---")
    print(full_text)  # Show first 500 characters
    # print("..." if len(full_text) > 500 else "")
    print("--- End Preview ---\n")
    
    # Define Dutch B-VCA diploma number patterns
    # B-VCA diplomas typically have format: 1511238.15063294 (7 digits, dot, 8 digits)
    # OCR may split the number with spaces
    patterns = [
        r"\b(\d{7})\s*\.\s*(\d{4})\s+(\d{3,4})\b",                                       # Split format with space
        r"\b(\d{7}\.\d{8})\b",                                                           # Exact format: 1511238.15063294
        r"\b(\d{6,7}\.\d{7,9})\b",                                                       # Flexible: 6-7 digits, dot, 7-9 digits
        r"(?:Naam|Name)\s*[:\-]?\s*([A-Z]\.?\s*[A-Za-z]+)",                              # Extracts name after "Naam:" or "Name:"
        r"(?:Geboortedatum|Date of birth)\s*[:\-]?\s*(\d{2}[-]\d{2}[-]\d{4})",          # Date of birth
        r"(?:Afgiftedatum|Date of issuance)\s*[:\-.\s~]*(\d{2}[-]\d{2}[-]\d{4})",        # Date of issuance
        r"\b(\d{7,12})\b"                                                                 # Fallback: any long number sequence
    ]

    # Try each pattern in order of priority
    for i, pattern in enumerate(patterns, 1):
        match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Handle tuple matches (for split diploma numbers)
            if match.groups() and len(match.groups()) >= 3 and i == 1:
                # Reconstruct diploma number with dot: 1511238.1506294
                diploma_number = f"{match.group(1)}.{match.group(2)}{match.group(3)}"
            elif match.groups():
                diploma_number = match.group(1)
            else:
                diploma_number = match.group(0)
                
            diploma_number = diploma_number.replace(' ', '')
            print(f"✓ Match found with pattern #{i}: {pattern}")
            print(f"✓ Extracted value: {diploma_number}")
            return diploma_number

    print("✗ No matching pattern found in the document text.")
    print("\nAvailable patterns tried:")
    for i, pattern in enumerate(patterns, 1):
        print(f"  {i}. {pattern}")
    return None

# --- Example Usage ---
# Use a real file path to test (e.g., download a sample PDF diploma)
file_to_process = "VCA 3.jpeg" 

diploma_id = partition_pdf_api(file_to_process)

if diploma_id:
    print(f"\nSUCCESS: Extracted Diploma ID: {diploma_id}")
else:
    print("\nFAILURE: Could not find Diploma ID.")
"""
Diploma Number Detection using Unstructured Client API

Requirements:
- Virtual environment (venv) with all dependencies installed
- .env file with UNSTRUCTURED_API_KEY set

To run:
    & "C:/Users/kpatel/Desktop/Text detection using EAST/venv/Scripts/python.exe" lang_unstr_run.py

Dependencies in venv:
- unstructured-client
- python-dotenv
- requests and other standard packages
"""

import re
import os
from typing import Optional
# from unstructured_client import unstructured_client
import unstructured_client 
from unstructured_client.models import shared
from unstructured_client.models.operations import PartitionRequest
from dotenv import load_dotenv
load_dotenv()
import mimetypes

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or 'application/octet-stream'

def partition_pdf_api(file_path: str) -> Optional[str]:
    """
    Processes a Dutch B-VCA diploma file using the Unstructured API and extracts the diploma number.
    Expects format like: 1511238.15063294
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    
    # Check if API key is set
    api_key = os.getenv("UNSTRUCTURED_API_KEY")
    if not api_key:
        print("Error: UNSTRUCTURED_API_KEY not found in environment variables")
        print("Please set it in your .env file")
        return None
    
    content_type = get_mime_type(file_path)
    print(f"Processing file: {file_path}")
    print(f"Content type: {content_type}")

    # --- Stage 1: Call Unstructured API for Text Extraction ---
    try:
        # Initialize client with API key
        client = unstructured_client.UnstructuredClient(
            api_key_auth=api_key
        )
        
        with open(file_path, "rb") as f:
            file_content = f.read()
            
            # Create a PartitionRequest with the file
            # Use hi_res strategy for better OCR on images with text
            req = PartitionRequest(
                partition_parameters=shared.PartitionParameters(
                    files=shared.Files(
                        content=file_content,
                        file_name=os.path.basename(file_path)
                    ),
                    strategy=shared.Strategy.HI_RES,
                    languages=["nld"],  # Dutch language code
                    ocr_languages=["nld"],  # OCR in Dutch
                )
            )
            
            print("Sending request to Unstructured API...")
            response = client.general.partition(request=req)

        # Check if response has elements
        if not hasattr(response, 'elements') or not response.elements:
            print("Warning: No elements returned from API")
            return None
            
        # Combine all elements into a single string for searching
        full_text = "\n".join([element.get("text", "") for element in response.elements])
        print(f"Extracted text length: {len(full_text)} characters")

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except PermissionError:
        print(f"Error: Permission denied when reading file: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred during Unstructured API call: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())
        return None

    # --- Stage 2: Regex Matching (Local and fast) ---
    
    if not full_text or full_text.strip() == "":
        print("Warning: Extracted text is empty")
        return None
    
    print("\n--- Extracted Text Preview ---")
    print(full_text)  # Show first 500 characters
    # print("..." if len(full_text) > 500 else "")
    print("--- End Preview ---\n")
    
    # Define Dutch B-VCA diploma number patterns
    # B-VCA diplomas typically have format: 1511238.15063294 (7 digits, dot, 8 digits)
    # OCR may split the number with spaces
    patterns = [
        r"\b(\d{7})\s*\.\s*(\d{4})\s+(\d{3,4})\b",                                       # Split format with space
        r"\b(\d{7}\.\d{8})\b",                                                           # Exact format: 1511238.15063294
        r"\b(\d{6,7}\.\d{7,9})\b",                                                       # Flexible: 6-7 digits, dot, 7-9 digits
        r"(?:Naam|Name)\s*[:\-]?\s*([A-Z]\.?\s*[A-Za-z]+)",                              # Extracts name after "Naam:" or "Name:"
        r"(?:Geboortedatum|Date of birth)\s*[:\-]?\s*(\d{2}[-]\d{2}[-]\d{4})",          # Date of birth
        r"(?:Afgiftedatum|Date of issuance)\s*[:\-.\s~]*(\d{2}[-]\d{2}[-]\d{4})",        # Date of issuance
        r"\b(\d{7,12})\b"                                                                 # Fallback: any long number sequence
    ]

    # Try each pattern in order of priority
    for i, pattern in enumerate(patterns, 1):
        match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
        if match:
            # Handle tuple matches (for split diploma numbers)
            if match.groups() and len(match.groups()) >= 3 and i == 1:
                # Reconstruct diploma number with dot: 1511238.1506294
                diploma_number = f"{match.group(1)}.{match.group(2)}{match.group(3)}"
            elif match.groups():
                diploma_number = match.group(1)
            else:
                diploma_number = match.group(0)
                
            diploma_number = diploma_number.replace(' ', '')
            print(f"✓ Match found with pattern #{i}: {pattern}")
            print(f"✓ Extracted value: {diploma_number}")
            return diploma_number

    print("✗ No matching pattern found in the document text.")
    print("\nAvailable patterns tried:")
    for i, pattern in enumerate(patterns, 1):
        print(f"  {i}. {pattern}")
    return None

# --- Example Usage ---
# Use a real file path to test (e.g., download a sample PDF diploma)
file_to_process = "VCA 3.jpeg" 

diploma_id = partition_pdf_api(file_to_process)

if diploma_id:
    print(f"\nSUCCESS: Extracted Diploma ID: {diploma_id}")
else:
    print("\nFAILURE: Could not find Diploma ID.")

