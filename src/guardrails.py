"""
guardrails.py
Safety and quality checks for HHGOA-SonicQuery.

Three checks, used in this order in the pipeline:
1. is_safe_query(question)                          — BEFORE anything else runs
2. should_answer(top_similarity_score)               — AFTER retrieval, BEFORE generation
3. is_grounded(answer_text, retrieved_chunks)         — AFTER generation, BEFORE showing the answer
"""

import time
from dotenv import load_dotenv
import os
from groq import Groq, APIStatusError, RateLimitError

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

client = Groq(api_key=groq_key) if groq_key else None


# ---------- 1. SAFE QUERY CHECK (runs first, no API call) ----------

# Basic keyword filter — catches obviously unsafe/off-topic queries before
# wasting a retrieval or generation call on them.
UNSAFE_KEYWORDS = [
    "kill", "suicide", "bomb", "weapon", "hack", "password",
    "credit card", "murder", "self harm", "explosive"
]

def is_safe_query(question):
    """
    Simple keyword filter. Returns False if the question contains any
    flagged keyword, True otherwise. This is a first line of defense,
    not a complete safety system.
    """
    question_lower = question.lower()
    for keyword in UNSAFE_KEYWORDS:
        if keyword in question_lower:
            return False
    return True


# ---------- 2. SHOULD ANSWER CHECK (runs after retrieval, no API call) ----------

def should_answer(top_similarity_score, threshold=0.5):
    """
    Returns False if the best-matching chunk's similarity score is below
    the threshold — meaning FAISS didn't find anything relevant enough,
    so we shouldn't bother calling Gemini at all.
    """
    return top_similarity_score >= threshold


# ---------- 3. GROUNDING CHECK (runs after generation, ONE extra Gemini call) ----------

def is_grounded(answer_text, retrieved_chunks, max_retries=3):
    """
    Asks Gemini a second time: is this answer actually supported by the
    retrieved context, or did it drift/hallucinate? Returns True only if
    Gemini's reply starts with "yes".

    Handles two distinct failure types differently:
    - errors.ClientError (429, quota exceeded) — free tier resets per
      minute, so this waits longer (35s) before retrying.
    - errors.ServerError (503, model overloaded) — usually clears fast,
      so this uses a short increasing backoff (5s, 10s, 20s).

    If all retries are exhausted, fails safe: treats the answer as NOT
    grounded rather than showing an unverified answer.
    """
    context = "\n\n".join(
        f"Chunk {i+1}: {chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    prompt = f"""Context:
{context}

Answer given: {answer_text}

Is this answer fully supported by the context above? Reply with only one word: yes or no."""

    server_error_waits = [5, 10, 20]

    if client is None:
        return False

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            reply = response.choices[0].message.content.strip().lower()
            return reply.startswith("yes")

        except RateLimitError as e:
            print(f"Attempt {attempt} failed (quota exceeded: {e}).")
            if attempt == max_retries:
                return False
            print("Waiting 35s for free-tier quota to reset...")
            time.sleep(35)

        except APIStatusError as e:
            if e.status_code < 500:
                return False
            print(f"Attempt {attempt} failed (server busy: {e}).")
            if attempt == max_retries:
                return False
            wait = server_error_waits[attempt - 1]
            print(f"Retrying in {wait}s...")
            time.sleep(wait)


if __name__ == "__main__":
    # Quick manual tests

    print("is_safe_query tests:")
    print(" 'What is the capital of France?' ->", is_safe_query("What is the capital of France?"))
    print(" 'How to make a bomb' ->", is_safe_query("How to make a bomb"))

    print("\nshould_answer tests:")
    print(" score 0.85, threshold 0.5 ->", should_answer(0.85))
    print(" score 0.3, threshold 0.5 ->", should_answer(0.3))

    print("\nis_grounded test:")
    sample_chunks = [
        {"text": "The Eiffel Tower is located in Paris, France and was completed in 1889."}
    ]
    good_answer = "The Eiffel Tower is in Paris, France."
    bad_answer = "The Eiffel Tower is in London, England."
    print(" grounded answer ->", is_grounded(good_answer, sample_chunks))
    print(" ungrounded answer ->", is_grounded(bad_answer, sample_chunks))