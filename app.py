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

st.set_page_config(page_title="HHGOA-SonicQuery")
st.title("HHGOA-SonicQuery")
st.write("Tap the mic, ask your question, tap again to stop recording.")


@st.cache_resource
def warm_up_retrieval():
    """
    Forces the embedding model (and FAISS index) to load once and stay
    cached for the whole app session, so real queries later don't pay the
    one-time model-loading cost — this is what keeps search() latency
    under 200ms.
    """
    search("warm up", top_k=1)
    return True


warm_up_retrieval()


def transcribe_audio(audio_bytes: bytes) -> str:
    """Sends recorded audio to Sarvam's speech-to-text API, returns the transcript."""
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
        question = None
        try:
            question = transcribe_audio(audio["bytes"])
        except requests.exceptions.RequestException:
            st.error(
                "Sorry, something went wrong while transcribing your audio. "
                "Please try again."
            )
        except Exception:
            st.error("An unexpected error occurred. Please try again.")

    if question:
        st.write(f"**You asked:** {question}")

        if not is_safe_query(question):
            st.warning("This question can't be processed. Please ask something else.")
        else:
            # Step 1 — search once, timed. Quick Answer's latency = this time only.
            t0 = time.perf_counter()
            retrieved_chunks = search(question, top_k=5)
            quick_latency_ms = (time.perf_counter() - t0) * 1000

            top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0

            # Step 2 — guardrail runs BEFORE generating either answer
            if not should_answer(top_score):
                st.warning("I don't have enough relevant information to answer that.")
            else:
                fast_answer = generate_fast_answer(retrieved_chunks)

                # Step 3 — same retrieved_chunks, timed separately from retrieval
                t1 = time.perf_counter()
                try:
                    polished_answer = generate_polished_answer(question, retrieved_chunks)
                except Exception:
                    polished_answer = (
                        "Answer service temporarily unavailable, please try again."
                    )
                polished_latency_ms = (time.perf_counter() - t1) * 1000

                # Step 4 — grounding check only on the polished answer
                grounded = is_grounded(polished_answer, retrieved_chunks)

                # Step 5 — display both, each labeled with its own latency
                st.markdown(f"**Quick Answer ({quick_latency_ms:.0f}ms):** {fast_answer}")
                st.markdown(f"**Polished Answer ({polished_latency_ms:.0f}ms):** {polished_answer}")
                if not grounded:
                    st.caption(
                        "⚠️ The polished answer may not be fully supported by the retrieved sources."
                    )