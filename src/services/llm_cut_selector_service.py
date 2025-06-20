# src/services/llm_cut_selector_service.py
import logging
import json
import requests
from typing import Dict, Any

class LLMCutSelectorService:
    """
    A service that uses an LLM to identify and tag segments for removal
    from a transcript.
    """
    def __init__(self, api_key: str):
        print("LLMCutSelectorService initialized.")
        if not api_key or "YOUR_LLM_API_KEY_HERE" in api_key:
            raise ValueError("LLM API key is not configured in config.yaml.")
        self.api_key = api_key
        self.url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_key}'

    def run(self, transcript: str) -> str:
        """
        Sends the transcript to the LLM and returns the marked-up version.

        Args:
            transcript: The full text transcript of the audio.

        Returns:
            The transcript with segments wrapped in <cut>...</cut> tags.
        """
        logging.info("Requesting cut selection from LLM...")
        
        # --- FIX: Restored the prompt to your original, complete version ---
        prompt = f"""
You are an expert audio editor functioning as a precise API.

Your task is to identify all filler words, repeated words, self-corrections, and verbal tics in the provided transcript. You will mark these segments for deletion.

**RULES:**
1.  You will mark segments for deletion by enclosing them in `<cut>` and `</cut>` tags.
2.  You **MUST NOT** alter, add, or remove any other part of the original text. The output must be the complete, original transcript, with only the addition of the `<cut>` tags.
3.  You **MUST** MARK at least one segment.
4.  Your response **MUST** contain **ONLY** the modified transcript text. Do not include any explanations, greetings, or markdown formatting.


**EXAMPLES:**

**Example 1: Simple filler word**
* **Original Transcript:** `So, um, I was thinking about the project.`
* **Your Response:** `So, <cut>um</cut>, I was thinking about the project.`

**Example 2: Multi-word filler phrase**
* **Original Transcript:** `And it was, you know, a very difficult decision.`
* **Your Response:** `And it was, <cut>you know</cut>, a very difficult decision.`

**Example 3: Stutter or repetition**
* **Original Transcript:** `I I think we should go with the first option.`
* **Your Response:** `<cut>I</cut> I think we should go with the first option.`

**Example 4: Self-correction or restart**
* **Original Transcript:** `We need to go to the... to the store.`
* **Your Response:** `We need to go <cut>to the...</cut> to the store.`

---

**FINAL TASK:**

Apply these rules to the following transcript. Remember to only return the modified text.

**Transcript:**
`{transcript}`
"""
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0}
        }

        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            response_json = response.json()
            marked_transcript = response_json['candidates'][0]['content']['parts'][0]['text']
            logging.info("Successfully received marked transcript from LLM.")
            return marked_transcript
        except requests.exceptions.RequestException as e:
            logging.error(f"API request to LLM failed: {e}", exc_info=True)
            raise
        except (KeyError, IndexError) as e:
            logging.error(f"Failed to parse LLM response: {e}", exc_info=True)
            logging.error(f"Raw response was: {response.text}")
            raise ValueError("Could not parse LLM response.")
        