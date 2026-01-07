import fitz  # PyMuPDF
from PIL import Image
import os

pdf_path = "VCABirek.pdf"
output_folder = "extracted_images_pymupdf"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

doc = fitz.open(pdf_path)  # Open the PDF document

for i in range(doc.page_count):
    page = doc.load_page(i)
    # Render page to a pixmap (image representation)
    # Use dpi parameter to set resolution (e.g., 300 for high quality)
    pix = page.get_pixmap(dpi=300)
    
    # Save the pixmap as an image file
    output_filename = f"{output_folder}/page_{i+1}.png"
    pix.save(output_filename)

doc.close()
print(f"Conversion complete. Images saved in '{output_folder}' folder.")
