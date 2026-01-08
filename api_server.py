"""
FastAPI Server for Dutch Certificate Data Extraction
Accepts PDF/Image uploads and returns structured JSON
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil
from pathlib import Path
import tempfile
from typing import Optional
import logging

# Import extraction logic for different document types
from diploma_data_extractor import process_file_with_vlm, CertificateData
from insurance_data_extractor import (
    process_insurance_file,
    process_insurance_file_with_debug,
    extract_insurance_data_vlm,
    InsuranceData,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
APP_REV = "ins-debug-tail-2025-12-15"

# Initialize FastAPI app
app = FastAPI(
    title="Dutch Certificate Data Extractor API",
    description="Upload PDF or Image files to extract structured certificate data",
    version="1.0.0"
)

# Add CORS middleware (allows frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB (increased for long insurance PDFs)

# Document types
DOCUMENT_TYPES = {
    0: "VCA Certificate",
    1: "Insurance Document",
    2: "Reserved for future",
    3: "Reserved for future"
}


def validate_file(filename: str, file_size: int) -> tuple[bool, Optional[str]]:
    """
    Validate uploaded file
    Returns (is_valid, error_message)
    """
    # Check file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f} MB"
    
    return True, None


@app.get("/")
async def root():
    """Root endpoint - Serve frontend HTML"""
    html_file = Path("frontend.html")
    if html_file.exists():
        with open(html_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    
    return {
        "message": "Dutch Certificate Data Extractor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "extract": "/extract (POST)",
            "docs": "/docs",
            "frontend": "/"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "certificate-extractor",
        "model": "Qwen3-VL-8B-Instruct",
        "revision": APP_REV
    }

@app.on_event("startup")
async def log_routes_on_startup():
    try:
        from fastapi.routing import APIRoute
        route_paths = [r.path for r in app.routes if isinstance(r, APIRoute)]
        print(f"Registered routes: {route_paths}")
    except Exception as e:
        logger.warning(f"Unable to list routes on startup: {e}")

@app.get("/version")
def version():
    return {"revision": APP_REV}

@app.post("/extract")
async def extract_document_data(
    file: UploadFile = File(...),
    document_type: int = Query(0, description="0=VCA, 1=Insurance"),
    debug: bool = Query(False, description="Include debug logs for troubleshooting (insurance)")
):
    """
    Extract structured data from document PDF/Image
    
    Args:
        file: PDF or Image file (max 50MB)
        document_type: Type of document (0=VCA Certificate, 1=Insurance, 2-3=Reserved)
    
    Returns:
        JSON with document data
    """
    temp_file_path = None
    debug_logs: list[str] = []  # Initialize at function scope so it's always available
    data_type = None
    
    try:
        # Validate document type
        if document_type not in DOCUMENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid document_type. Allowed values: {list(DOCUMENT_TYPES.keys())}"
            )
        
        # Log request
        print(f"Received file: {file.filename}, document_type: {document_type} ({DOCUMENT_TYPES[document_type]})")
        
        # Read file content to check size
        file_content = await file.read()
        file_size = len(file_content)
        
        # Validate file
        is_valid, error_msg = validate_file(file.filename, file_size)
        if not is_valid:
            logger.warning(f"File validation failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        print(f"File saved to: {temp_file_path} ({file_size} bytes)")
        
        # Route to appropriate processor based on document type
        print(f"Processing as: {DOCUMENT_TYPES[document_type]}")
        
        if document_type == 0:
            # VCA Certificate processing
            print("Starting VCA certificate processing...")
            extracted_data = process_file_with_vlm(temp_file_path)
            data_type = "certificate"
            
        elif document_type == 1:
            # Insurance document processing
            print("Starting insurance document processing...")
            extracted_data = extract_insurance_data_vlm(
                temp_file_path, max_pages=50, debug=debug_logs
            )
            data_type = "insurance"
            
        elif document_type in [2, 3]:
            # Future document types
            raise HTTPException(
                status_code=501,
                detail=f"Document type {document_type} is reserved for future implementation"
            )
        
        if extracted_data is None:
            logger.error("Document processing returned None")
            # Include debug logs if available for insurance
            if document_type == 1 and debug_logs:
                truncated = "\n".join(debug_logs[-80:])
                # Friendly hint when no API key is present
                hint = ""
                if any("No API key found" in line for line in debug_logs):
                    hint = ("\n\nHint: No HF/OpenAI token detected. Set one of: "
                            "HF_TOKEN, HUGGINGFACEHUB_API_TOKEN, or OPENAI_API_KEY.")
                if truncated:
                    logger.error("Insurance extraction debug tail:\n" + truncated)
                raise HTTPException(
                    status_code=500,
                    detail={
                        "message": f"Failed to extract {DOCUMENT_TYPES[document_type]} data.",
                        "debug_tail": truncated,
                        "hint": hint.strip() if hint else "",
                        "revision": APP_REV,
                    }
                )
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Failed to extract {DOCUMENT_TYPES[document_type]} data. "
                    f"Please ensure the file contains a valid document."
                )
            )
        
        # Convert to dict
        result = extracted_data.model_dump()
        
        print(f"Successfully extracted {data_type} data")
        
        # Build response
        response_data = {
            "success": True,
            "document_type": document_type,
            "document_type_name": DOCUMENT_TYPES[document_type],
            "data": result,
            "filename": file.filename,
            "revision": APP_REV,
        }
        
        # Always include debug logs for insurance documents if available
        if document_type == 1 and debug_logs:
            response_data["debug"] = debug_logs
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
    

        
    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@app.post("/extract-batch")
async def extract_certificate_data_batch(files: list[UploadFile] = File(...)):
    """
    Extract structured data from multiple certificate files
    
    Args:
        files: List of PDF or Image files (max 10MB each)
    
    Returns:
        JSON with array of certificate data
    """
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch request"
        )
    
    results = []
    temp_files = []
    
    try:
        for file in files:
            try:
                print(f"Processing batch file: {file.filename}")
                
                # Read and validate file
                file_content = await file.read()
                file_size = len(file_content)
                
                is_valid, error_msg = validate_file(file.filename, file_size)
                if not is_valid:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": error_msg,
                        "data": None
                    })
                    continue
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name
                    temp_files.append(temp_file_path)
                
                # Process with VLM
                certificate_data = process_file_with_vlm(temp_file_path)
                
                if certificate_data:
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "error": None,
                        "data": certificate_data.model_dump()
                    })
                else:
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": "Failed to extract data",
                        "data": None
                    })
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": str(e),
                    "data": None
                })
        
        successful = sum(1 for r in results if r["success"])
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "total_files": len(files),
                "successful": successful,
                "failed": len(files) - successful,
                "results": results
            }
        )
    
    finally:
        # Cleanup all temp files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_path}: {e}")


@app.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats and document types"""
    return {
        "supported_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
        "document_types": DOCUMENT_TYPES
    }

@app.get("/document-types")
async def get_document_types():
    """Get list of supported document types"""
    return {
        "document_types": DOCUMENT_TYPES,
        "usage": {
            "description": "Use document_type parameter in /extract endpoint",
            "example": "POST /extract?document_type=1 (for insurance documents)"
        }
    }


if __name__ == "__main__":
    # Run the server
    print("="*70)
    print("🚀 Starting Dutch Certificate Extractor API Server")
    print("="*70)
    port = int(os.getenv("PORT", "8000"))
    print(f"📍 Server URL: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print(f"📊 Health Check: http://localhost:{port}/health")
    print("="*70)
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )