from dotenv import load_dotenv   # ← pulls the tool out of the toolbox
import os
import requests
load_dotenv()                     # ← actually uses the tool: reads your .env file
sarvam_key = os.getenv("SARVAM_API_KEY")   # ← grabs the specific key you need



url = "https://api.sarvam.ai/speech-to-text"
headers = {"api-subscription-key": sarvam_key}
files = {"file": ("test.wav", open("test.wav", "rb"), "audio/wav")}
data = {"model": "saaras:v3", "mode": "transcribe"}

response = requests.post(url, headers=headers, files=files, data=data)

if response.status_code == 200:
    result = response.json()
    print("Transcript:", result.get("transcript"))
else:
    print("Error:", response.status_code, response.text)