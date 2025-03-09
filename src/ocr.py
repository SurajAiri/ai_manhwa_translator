import easyocr
import cv2
import numpy as np

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Bilateral Filtering (Removes noise while keeping edges)
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive Thresholding (better than Otsu in uneven lighting)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=15, C=3)

    # Morphological Operations to remove small specks
    kernel = np.ones((1,1), np.uint8)  # Smaller kernel to avoid affecting text
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    return cleaned


def extract_text_from_images(cropped_images):
    # Initialize EasyOCR reader for the desired languages
    reader = easyocr.Reader(['ko'])  # Japanese, Korean, Chinese

    extracted_texts = []
    
    for cropped in cropped_images:
        # Perform OCR
        results = reader.readtext(process_image(cropped['image']), detail=0, paragraph=True)  # detail=0 returns only text
        
        # Join the results into a single string
        text = " ".join(results).strip()
        
        # Store the extracted text
        extracted_texts.append({
            "id": cropped['id'],
            "text": text,  # Clean up the text
            "bbox": cropped['bbox'],
            "path": cropped['path']
        })
    
    return extracted_texts
