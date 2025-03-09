
from src.text_area import detect_bubbles_with_yolo
from src.crop_image import crop_text_regions
from src.ocr import extract_text_from_images
from src.translate import Translator
from src.text_overlay import inpaint_text, overlay_text
import pyperclip
import cv2
from src.utility import parse_translation


# Initialize the translator
translator = Translator()
image_path = "samples/sample4.jpg"
output = cv2.imread(image_path)
translation_model = "manual" # openai, manual, gemini (default)
is_debug = False

# Path to the image
output_debug, result = detect_bubbles_with_yolo(image_path)
print("detect bubble: ",result)

# filter the bubbles
bubbles = [bubble for bubble in result if bubble['confidence'] > 0.5]

# cropping image
cropped_texts = crop_text_regions(output, bubbles, is_debug=is_debug)
# print("cropped text: ",cropped_texts)

# extract text from images
extracted_texts = extract_text_from_images(cropped_texts)
print("extracted text: ",extracted_texts)

# translate the text
texts = [text_info['text'] for text_info in extracted_texts]
if translation_model == "manual":
    prompt = translator.manual_translate_prompt(texts)
    # copy the prompt to the clipboard
    pyperclip.copy(prompt)
    # wait for user input to continue
    input("Prompt copied to clipboard. Press Enter to continue...")
    translated_texts = parse_translation(input("Enter the translated text (Paste AI response): "))
else:
    translated_texts = parse_translation(translator.translate_text(texts, model=translation_model))
print("translated text: ",translated_texts)

if len(translated_texts) != len(extracted_texts):
    raise ValueError("Number of translated texts does not match the number of extracted texts.")

# update translated text
for i, text_info in enumerate(extracted_texts):
    text_info['translated_text'] = translated_texts[i]

print("after translation, extracted text: ",extracted_texts)

# Display the debug image
if is_debug:
    cv2.imshow("Debug", output_debug)


# extracted_texts = [{'id': 0, 'text': '흙. 그럼 대호는 벌써 옷올 갈아입고 다른 곳으로 갖나  ?', 'bbox': (468, 79, 954, 588), 'path': 'cropped_texts/bubble_0.png', 'translated_text': 'Dirt. So, Daeho already changed clothes and went somewhere else?'}]


# merge on the image
output = overlay_text(output, extracted_texts)

# save the result
cv2.imwrite("output.png", output)

# print mouse position when left mouse button is clicked
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

cv2.imshow("Result", output)
cv2.setMouseCallback("Result", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
