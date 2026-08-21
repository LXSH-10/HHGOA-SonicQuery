import os
import time

import requests
import streamlit as st
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

from src.retrieval import search
from src.generation import generate_fast_answer, generate_polished_answer
from src.guardrails import should_answer, is_grounded, is_safe_query

load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_URL = "https://api.sarvam.ai/speech-to-text"

# Maps Sarvam's BCP-47 codes to the "language" tag stored in chunks.json,
# and to a natural-language name for instructing Gemini.
LANGUAGE_INFO = {
    "hi-IN": {"tag": "hindi", "name": "Hindi"},
    "mr-IN": {"tag": "marathi", "name": "Marathi"},
    "en-IN": {"tag": "english", "name": "English"},
}

st.set_page_config(page_title="HHGOA-SonicQuery")
st.title("HHGOA-SonicQuery")
st.write("Tap the mic, ask your question, tap again to stop recording.")


@st.cache_resource
def warm_up_retrieval():
    """Forces the embedding model + FAISS index to load once, cached for the session."""
    search("warm up", top_k=1)
    return True


warm_up_retrieval()


def transcribe_audio(audio_bytes: bytes):
    """
    Sends recorded audio to Sarvam's speech-to-text API. Returns
    (transcript, language_code) — language_code comes from Sarvam's
    auto-detection since we don't specify one in the request.
    """
    files = {"file": ("recording.wav", audio_bytes, "audio/wav")}
    data = {"model": "saaras:v3", "mode": "transcribe"}
    headers = {"api-subscription-key": SARVAM_API_KEY}

    response = requests.post(
        SARVAM_URL, headers=headers, files=files, data=data, timeout=30
    )
    response.raise_for_status()
    result = response.json()
    return result.get("transcript", ""), result.get("language_code", "")


def extract_fast_answer_text(fast_answer):
    """
    generate_fast_answer() returns a dict like {"answer": ..., "similarity_score": ...}.
    This pulls out just the answer text so it displays cleanly instead of
    printing the raw dict.
    """
    if isinstance(fast_answer, dict):
        return fast_answer.get("answer", str(fast_answer))
    return fast_answer


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
        question, lang_code = None, ""
        try:
            question, lang_code = transcribe_audio(audio["bytes"])
        except requests.exceptions.RequestException:
            st.error(
                "Sorry, something went wrong while transcribing your audio. "
                "Please try again."
            )
        except Exception:
            st.error("An unexpected error occurred. Please try again.")

    if question:
        lang_info = LANGUAGE_INFO.get(lang_code)
        lang_tag = lang_info["tag"] if lang_info else None
        lang_name = lang_info["name"] if lang_info else None

        st.write(f"**You asked:** {question}")
        if lang_name:
            st.caption(f"Detected language: {lang_name}")

        if not is_safe_query(question):
            st.warning("This question can't be processed. Please ask something else.")
        else:
            # Step 1 — search, filtered to the detected language if known
            t0 = time.perf_counter()
            retrieved_chunks = search(question, top_k=5, language=lang_tag)
            quick_latency_ms = (time.perf_counter() - t0) * 1000

            top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0

            if not should_answer(top_score):
                st.warning("I don't have enough relevant information to answer that.")
            else:
                fast_answer = generate_fast_answer(retrieved_chunks)
                fast_answer_text = extract_fast_answer_text(fast_answer)

                t1 = time.perf_counter()
                try:
                    polished_answer = generate_polished_answer(
                        question, retrieved_chunks, language_name=lang_name
                    )
                except Exception:
                    polished_answer = (
                        "Answer service temporarily unavailable, please try again."
                    )
                polished_latency_ms = (time.perf_counter() - t1) * 1000

                grounded = is_grounded(polished_answer, retrieved_chunks)

                st.markdown(f"**Quick Answer ({quick_latency_ms:.0f}ms):** {fast_answer_text}")
                st.markdown(f"**Polished Answer ({polished_latency_ms:.0f}ms):** {polished_answer}")
                if not grounded:
                    st.caption(
                        "⚠️ The polished answer may not be fully supported by the retrieved sources."
                    )