"""
Test client for Certificate Extractor API
"""

import requests
import json
from pathlib import Path

# API base URL
API_URL = "http://localhost:8000"


def test_health_check():
    """Test health check endpoint"""
    print("\n" + "="*70)
    print("Testing Health Check")
    print("="*70)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    

def test_extract_single_file(file_path: str, document_type: int = 0):
    """Test single file extraction
    
    Args:
        file_path: Path to the document file
        document_type: 0=VCA Certificate, 1=Insurance Document, 2-3=Reserved
    """
    doc_types = {0: "VCA Certificate", 1: "Insurance Document", 2: "Reserved", 3: "Reserved"}
    
    print("\n" + "="*70)
    print(f"Testing Single File Extraction: {Path(file_path).name}")
    print(f"Document Type: {document_type} ({doc_types.get(document_type, 'Unknown')})")
    print("="*70)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    # Open and send file with document_type parameter
    with open(file_path, 'rb') as f:
        files = {'file': (Path(file_path).name, f, 'application/octet-stream')}
        
        response = requests.post(
            f"{API_URL}/extract?document_type={document_type}",
            files=files
        )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ SUCCESS!")
        print(json.dumps(result, indent=2))
    else:
        print(f"\n❌ ERROR!")
        print(response.text)


def test_extract_batch(file_paths: list):
    """Test batch file extraction"""
    print("\n" + "="*70)
    print(f"Testing Batch Extraction ({len(file_paths)} files)")
    print("="*70)
    
    files_data = []
    for file_path in file_paths:
        if Path(file_path).exists():
            files_data.append(
                ('files', (Path(file_path).name, open(file_path, 'rb'), 'application/octet-stream'))
            )
        else:
            print(f"⚠️  Skipping missing file: {file_path}")
    
    if not files_data:
        print("❌ No valid files to process")
        return
    
    response = requests.post(
        f"{API_URL}/extract-batch",
        files=files_data
    )
    
    # Close file handles
    for _, (_, file_obj, _) in files_data:
        file_obj.close()
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ BATCH RESULTS:")
        print(json.dumps(result, indent=2))
    else:
        print(f"\n❌ ERROR!")
        print(response.text)


def test_supported_formats():
    """Test supported formats endpoint"""
    print("\n" + "="*70)
    print("Testing Supported Formats")
    print("="*70)
    
    response = requests.get(f"{API_URL}/supported-formats")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_document_types():
    """Test document types endpoint"""
    print("\n" + "="*70)
    print("Testing Document Types")
    print("="*70)
    
    response = requests.get(f"{API_URL}/document-types")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    print("="*70)
    print("🧪 API CLIENT TEST SUITE")
    print("="*70)
    
    # Test 1: Health check
    test_health_check()
    
    # Test 2: Supported formats
    test_supported_formats()
    
    # Test 3: Document types
    test_document_types()
    
    # Test 4: Single file extraction - VCA Certificate (Type 0)
    test_extract_single_file(r"input_data\17374835591093552603982971278860.jpg", document_type=0)
    
    # Test 5: Single file extraction - VCA PDF (Type 0)
    test_extract_single_file(r"input_data\VCA 1.pdf", document_type=0)
    
    # Test 6: Single file extraction - Insurance Document (Type 1)
    # test_extract_single_file(r"Insurance_data\sample_insurance.pdf", document_type=1)
    
    # Test 5: Batch extraction
    batch_files = [
        r"input_data\17374835591093552603982971278860.jpg",
        r"input_data\VCA 1.pdf",
        r"input_data\VCA 3.jpeg"
    ]
    test_extract_batch(batch_files)
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70)
