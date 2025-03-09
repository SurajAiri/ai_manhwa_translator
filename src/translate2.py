from transformers import MarianMTModel, MarianTokenizer

def translate_text(text, src_lang='ja', tgt_lang='en'):
    # Specify the model name based on the source and target languages
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    
    # Load the tokenizer and model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Prepare the text for translation
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True, truncation=True))
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated_text

# Example usage
text_to_translate = "흙. 그럼 대호는 벌써 옷올 갈아입고 다른 곳으로 갖나  ?"  # "Hello" in Japanese
translated_text = translate_text(text_to_translate, src_lang='ko', tgt_lang='en')
print(f"Translated: {translated_text}")


import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()



# Configure the Gemini API
GOOGLE_API_KEY = os.environ['GEMINI_API_KEY']  # Replace with your actual API key (get from Google AI Studio)
print(GOOGLE_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-1.5-flash')  # Use Gemini Flash 2.0

def translate_text(text, lang='korean'):
    """Translates text using the Gemini API."""
    try:

        prompt = f"{text} translate {lang} manhwa dialogue to english. Don't return anything else."
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Translation Error: {e}")
        return None

korean_text = "흙. 그럼 대호는 벌써 옷올 갈아입고 다른 곳으로 갖나  ?"  

if korean_text:
    english_translation = translate_text(korean_text)
    if english_translation:
        print("English Translation:")
        print(english_translation)
    else:
        print("Translation failed.")