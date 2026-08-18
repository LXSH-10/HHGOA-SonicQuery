from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_key)

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents="What is the capital of France?"
)

print(response.text)