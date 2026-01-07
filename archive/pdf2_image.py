import os
import fitz  # PyMuPDF

def pdf_to_images(pdf_path, output_folder='test'):
    """Convert PDF to images using PyMuPDF (no Poppler needed!)"""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Open PDF
        pdf_document = fitz.open(pdf_path)
        
        # Iterate through pages
        for page_num in range(len(pdf_document)):
            # Get page
            page = pdf_document[page_num]
            
            # Convert to image (higher DPI = better quality)
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
            
            # Save image
            image_path = os.path.join(output_folder, f'page_{page_num + 1}.png')
            pix.save(image_path)
            print(f'✓ Saved: {image_path} ({pix.width}x{pix.height})')
        
        pdf_document.close()
        print(f'\n✅ Successfully converted {len(pdf_document)} pages!')
        
    except Exception as e:
        print(f'❌ Error: {e}')

if __name__ == "__main__":
    pdf_file = r'input_data\img20250911_17131927.pdf'
    pdf_to_images(pdf_file)
