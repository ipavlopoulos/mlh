import requests


class GeminiService:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.model = model

    @property
    def configured(self):
        return bool(self.api_key)

    def generate_content(self, payload):
        if not self.configured:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        response = requests.post(
            url,
            params={"key": self.api_key},
            json=payload,
            timeout=120,
        )
        if not response.ok:
            raise RuntimeError(f"Gemini API request failed with status {response.status_code}")
        return response.json()
