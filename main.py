
from src.text_area import detect_bubbles_with_yolo
from src.crop_image import crop_text_regions
from src.ocr import extract_text_from_images
import cv2

# Path to the image
image_path = "samples/sample1.png"
output, result = detect_bubbles_with_yolo(image_path)
print(result)


# cropping image

# Example usage
# result_image, text_regions = detect_bubbles_with_yolo(image_path)
cropped_texts = crop_text_regions(output, result)

extracted_texts = extract_text_from_images(cropped_texts)

print(extracted_texts)
# Print extracted texts
for text_info in extracted_texts:
    print(f"Bubble ID: {text_info['id']}, Text: {text_info['text']}")

# translate the text

# merge on the image

# Display the result
# print(result)
# output = cv2.rectangle(output, bubble_region[0], bubble_region[1], (0, 255, 0), 2)
cv2.imshow("Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
