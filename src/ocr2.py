import cv2
import numpy as np
import easyocr

def preprocess_image(image_path):
    """
    Preprocess the image by converting it to grayscale, enhancing contrast,
    applying adaptive thresholding, denoising, and morphological operations.
    """
    # Read image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(thresh, h=30)

    # Morphological Operations
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
    
    return processed

def korean_ocr(image_path):
    """
    Perform OCR on a preprocessed image using EasyOCR for Korean language.
    """
    processed_image = preprocess_image(image_path)
    reader = easyocr.Reader(['ko'], gpu=True)
    results = reader.readtext(processed_image, detail=0)  # detail=0 for plain text output
    
    return results

# Example usage
if __name__ == "__main__":
    image_path = "../samples/sample1.png"  # Replace with actual image path
    text_output = korean_ocr(image_path)
    print("Extracted Text:", text_output)


import easyocr
import cv2

def easyocr_korean(image_path):
    reader = easyocr.Reader(['ko'], gpu=True)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # plt.imshow(thresh, cmap='gray')
    result = reader.readtext(thresh, detail=0)
    return " ".join(result)


image_path = "../cropped_texts/test.png"
text_output = easyocr_korean(image_path)
print("Extracted Text:", text_output)

import easyocr
import cv2
import numpy as np

from matplotlib import pyplot as plt

def easyocr_korean(image_path):
    reader = easyocr.Reader(['ko'], gpu=True)

    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)

    # Adaptive Thresholding (better than Otsu in some cases)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Denoising
    denoised = cv2.fastNlMeansDenoising(thresh, h=10)

    # Slight Morphological Operation to remove specks
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    plt.imshow(processed, cmap='gray')
    # OCR
    result = reader.readtext(processed, detail=0)
    
    return " ".join(result)

# Example usage
image_path = "../cropped_texts/bubble_0.png"
text_output = easyocr_korean(image_path)
print("Extracted Text:", text_output)


import easyocr
import cv2
import numpy as np

def easyocr_korean(image_path,lang_list=['ko']):
    reader = easyocr.Reader(lang_list, gpu=True)

    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Bilateral Filtering (Removes noise while keeping edges)
    filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Adaptive Thresholding (better than Otsu in uneven lighting)
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, blockSize=15, C=3)

    # Morphological Operations to remove small specks
    kernel = np.ones((1,1), np.uint8)  # Smaller kernel to avoid affecting text
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # OCR Processing
    result = reader.readtext(cleaned, detail=0)

    return " ".join(result)

# Example usage
image_path = "../cropped_texts/test2.png"
text_output = easyocr_korean(image_path)
print("Extracted Text:", text_output)

# 3: 아아 이거 너무 스길 넘처서 참지 못하켓어..
# 2: 으응 그래 어쩌면 다시 고기 잡으러 값흘지도.. ?
# 1: 흙. . 그럼 대호는 벌써 옷올 갈아입고 다른 곳으로 갖나  ?