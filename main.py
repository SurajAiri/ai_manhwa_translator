
from src.text_area import detect_bubbles_with_yolo
from src.crop_image import crop_text_regions
from src.ocr import extract_text_from_images
from src.translate import Translator
from src.text_overlay import inpaint_text, overlay_text
import cv2


# Initialize the translator
translator = Translator()
image_path = "samples/sample1.png"
output = cv2.imread(image_path)

# # Path to the image
# output, result = detect_bubbles_with_yolo(image_path)
# print("detect bubble: ",result)

# # filter the bubbles
# bubbles = [bubble for bubble in result if bubble['confidence'] > 0.5]


# # cropping image
# cropped_texts = crop_text_regions(output, bubbles)

# extracted_texts = extract_text_from_images(cropped_texts)

# print("extracted text: ",extracted_texts)
# # # Print extracted texts
# # for text_info in extracted_texts:
# #     print(f"Bubble ID: {text_info['id']}, Text: {text_info['text']}")

# # translate the text
# texts = [text_info['text'] for text_info in extracted_texts]
# translated_texts = translator.translate_text(texts)
# print("translated text: ",translated_texts)

# # update translated text
# for i, text_info in enumerate(extracted_texts):
#     text_info['translated_text'] = translated_texts['data'][i]

# print("after translation, extracted text: ",extracted_texts)

extracted_texts = [{'id': 0, 'text': '흙. 그럼 대호는 벌써 옷올 갈아입고 다른 곳으로 갖나  ?', 'bbox': (468, 79, 954, 588), 'path': 'cropped_texts/bubble_0.png', 'translated_text': 'Dirt. So, Daeho already changed clothes and went somewhere else?'}]
# merge on the image
# output = inpaint_text(output, extracted_texts)
output = overlay_text(output, extracted_texts)


# Display the result
# print(result)
# output = cv2.rectangle(output, bubble_region[0], bubble_region[1], (0, 255, 0), 2)
cv2.imshow("Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
