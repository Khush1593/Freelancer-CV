import cv2
import numpy as np

# ---------------------------
# 1. Load image
# ---------------------------
img_name = r"C:\Users\kpatel\Desktop\Text detection using EAST\input_data\Screenshot_20200127-163831_Chrome.jpg"

img = cv2.imread(img_name)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------------------------------------------------
# 2. SHADOW REMOVAL (Background estimation + subtraction)
# ---------------------------------------------------------
# Large blur to estimate background (shadow layer)
blur = cv2.GaussianBlur(gray, (55, 55), 0)

# # Subtract background from grayscale
no_shadow = cv2.divide(gray, blur, scale=255)

# ---------------------------------------------------------
# 3. LOCAL CONTRAST (CLAHE)
# ---------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(no_shadow)

# ---------------------------------------------------------
# 4. ADAPTIVE THRESHOLDING (Best for uneven lighting)
# ---------------------------------------------------------
binary = cv2.adaptiveThreshold(
    enhanced,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    25,           # block size
    9             # C value (fine tune if needed)
)

# ---------------------------------------------------------
# 5. Save output for OCR
# ---------------------------------------------------------
cv2.imwrite(r"output_no_shadow.png", no_shadow)
cv2.imwrite(r"output_contrast.png", enhanced)
cv2.imwrite(r"output_binary.png", binary)

print("Images are saved in the folder...")