from dotenv import load_dotenv   # ← pulls the tool out of the toolbox
import os

load_dotenv()                     # ← actually uses the tool: reads your .env file
sarvam_key = os.getenv("SARVAM_API_KEY")   # ← grabs the specific key you need
