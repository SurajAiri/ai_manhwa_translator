import json
import re


def parse_llm_json(text):
    """Parses the manual translation text and returns a list of translated text."""
    # Clean up markdown code block syntax if present
    # Remove markdown code blocks and any extra text before/after JSON
    text = re.sub(r'```json\s*|\s*```', '', text)
    # Extract only the JSON part (from first { to last })
    json_match = re.search(r'(\{.*\})', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    # convert text to json
    try:
        return json.loads(text)
    except Exception as e:
        print(f"Error parsing manual translation: {e}")
        return None
    
def parse_translation(text):
    """Parses the translation text and returns a list of translated text."""
    
    try:
        res = parse_llm_json(text)
        
        if not res or not isinstance(res.get('data'), list):
            raise ValueError("Invalid input: 'data' field must be a list.")
        
        return res['data']
        
    except Exception as e:
        raise ValueError(f"Error parsing translation text: {e}. Please make sure the text is in JSON format.")


# # example use
# raw = """ hi this is just a test: {"data": ["Dirt. So, Daeho already changed clothes and went somewhere else?"]}"""
# translated_texts = parse_translation(raw)
# print(translated_texts)  # ['Dirt. So, Daeho already changed clothes and went somewhere else?']