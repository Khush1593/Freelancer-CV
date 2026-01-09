# Project README

This project is for text detection using the EAST model and related document/image processing for insurance and VCA data extraction.

## Structure

- `archive/` - Old or experimental code, not actively used
- `Insurance_data/` - Insurance-related documents and images
- `VCA_Data/` - VCA certificates and related files

## Main Files
- `api_server.py` - Main API server
- `auth_utils.py` - Authentication utilities
- `hf_text_call.py` - HuggingFace text extraction     (WHERE)
- `insurance_data_extractor.py` - Insurance data extraction logic
- `pdf2_image.py` - PDF to image conversion
- `VLM_image_call.py` - Vision-Language Model image call   (WHERE)
- `frontend.html` - Frontend interface
- `test_api_client.py` - API client for testing

## How to Run
1. Set up your Python environment (see `venv/` or create a new one)
2. Install dependencies from `requirements.txt`
3. Run the API server: `python src/api_server.py`

## Notes
- Place new scripts in `src/`.
- Move deprecated/unused scripts to `archive/`.
- Store data files in `Insurance_data/` or `VCA_Data/` as appropriate.

## python version
Python 3.11.9