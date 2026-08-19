import os

import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

SARVAM_URL = "https://api.sarvam.ai/speech-to-text"

st.set_page_config(page_title="HHGOA-SonicQuery")
st.title("HHGOA-SonicQuery")
st.write("Tap the mic, ask your question, tap again to stop recording.")


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Sends recorded audio bytes to Sarvam's speech-to-text API and returns
    the transcript. Raises requests.exceptions.RequestException on failure,
    which the caller catches to show a friendly message.
    """
    files = {"file": ("recording.wav", audio_bytes, "audio/wav")}
    data = {"model": "saaras:v3", "mode": "transcribe"}
    headers = {"api-subscription-key": SARVAM_API_KEY}

    response = requests.post(
        SARVAM_URL, headers=headers, files=files, data=data, timeout=30
    )
    response.raise_for_status()
    return response.json().get("transcript", "")


audio = mic_recorder(
    start_prompt="🎤 Start recording",
    stop_prompt="⏹ Stop recording",
    just_once=True,
    use_container_width=True,
    key="recorder",
)

if audio is not None:
    st.audio(audio["bytes"])

    with st.spinner("Transcribing..."):
        try:
            transcript = transcribe_audio(audio["bytes"])
            if transcript:
                st.success("Transcribed successfully")
                st.write(f"**You said:** {transcript}")
            else:
                st.warning("No speech detected — please try again.")
        except requests.exceptions.RequestException:
            st.error(
                "Sorry, something went wrong while transcribing your audio. "
                "Please check your connection and try again."
            )
        except Exception:
            st.error("An unexpected error occurred. Please try again.")