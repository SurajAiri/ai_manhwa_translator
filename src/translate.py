from dotenv import load_dotenv
import os

load_dotenv()

class Translator:
    PROMPT = "{texts} translate {lang} manhwa dialogue to english.if translation not possible keep \"\" as placeholder to maintain order of list  Don't return anything else. return only list of translated text. in json format {{\"data\":[\"translation1\",\"translation2\"]}}"

    def __translate_gemini(self, texts ,model_name = "gemini-1.5-flash", lang='korean'):
        """Translates text using the Gemini API."""
        try:
            import google.generativeai as genai

            GOOGLE_API_KEY = os.environ['GEMINI_API_KEY']
            genai.configure(api_key=GOOGLE_API_KEY)
            model = genai.GenerativeModel(model_name if model_name else "gemini-1.5-flash")  

            prompt = self.PROMPT.format(lang=lang,texts=texts)
            print(prompt)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Translation Error: {e}")
            return None
        
    """not tested this don't recommend to use"""
    def __translate_huggingface(self, text, src_lang='ko'):
        try:
            from transformers import MarianMTModel, MarianTokenizer
            # Specify the model name based on the source and target languages
            model_name = f'Helsinki-NLP/opus-mt-{src_lang}-en'
            
            # Load the tokenizer and model
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            # Prepare the text for translation
            translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True, truncation=True))
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

            return translated_text
        except Exception as e:
            print(f"Hugging Face Translation Error: {e}")
            return None
        
    
    def __translate_openai(self, texts, model="gpt-4o-mini", lang='korean'):
        try:
            import openai
            client = openai.Client()
            prompt = self.PROMPT.format(lang=lang, texts=texts)
            response = client.chat.completions.create(
                model=model if model else "gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            print(response)
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI Translation Error: {e}")
            return None

    def manual_translate_prompt(self, texts, lang='korean'):
        return self.PROMPT.format(lang=lang, texts=texts)
    
    def translate_text(self, texts,model='gemini', lang='korean',model_name=None, src_lang='ko'):
        print(model)
        if model == 'huggingface':
            return self.__translate_huggingface(texts, src_lang)
        elif model == 'openai':
            return self.__translate_openai(texts, model_name,lang=lang)
        elif model == 'gemini': # default
            return self.__translate_gemini(texts, model_name,lang)
        elif model == 'manual':
            return self.manual_translate_prompt(texts, lang)
        else:
            raise ValueError(f"Unsupported model: {model}")
        

# example use
# 3: 아아 이거 너무 스길 넘처서 참지 못하켓어..
# 2: 으응 그래 어쩌면 다시 고기 잡으러 값흘지도.. ?
# 1: 흙. . 그럼 대호는 벌써 옷올 갈아입고 다른 곳으로 갖나  ?

# raw = [
#     '흙. . 그럼 대호는 벌써 옷올 갈아입고 다른 곳으로 갖나  ?',
#     '으응 그래 어쩌면 다시 고기 잡으러 값흘지도.. ?',
#     '아아 이거 너무 스길 넘처서 참지 못하켓어..'
# ]

# translator = Translator()
# translations = translator.translate_text(raw,model='openai')
