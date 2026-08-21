from dotenv import load_dotenv
import os
import time
from google import genai
from google.genai import errors

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_key)


def generate_fast_answer(retrieved_chunks):
    """
    No API call. Instantly returns the top-scoring chunk's text and its
    similarity score. Use this to show a near-instant raw answer while
    generate_polished_answer is still being generated.
    """
    top_chunk = retrieved_chunks[0]
    return {
        "answer": top_chunk["text"],
        "similarity_score": top_chunk["score"]
    }


def generate_polished_answer(question, retrieved_chunks, language_name=None, max_retries=3):
    """
    Calls Gemini and instructs it to answer ONLY using the given chunks.
    If the answer isn't present in them, it returns the exact fallback
    message instead of guessing.

    language_name: optional, e.g. "Hindi", "Marathi", "English". When given,
    tells Gemini to respond in that language. When None, no language
    instruction is added (unchanged behavior).

    Retries automatically on two distinct failure types:
    - errors.ServerError (503, model temporarily overloaded) — short
      increasing backoff, usually clears within seconds.
    - errors.ClientError (429, quota exceeded) — free tier limits reset
      per minute, so this waits longer before retrying.

    Fully independent from generate_fast_answer so the app can time
    and display each one separately.
    """
    # Join all chunk texts into one labeled context block
    context = "\n\n".join(
        f"Chunk {i+1}: {chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    language_instruction = (
        f"\nRespond in {language_name} only.\n" if language_name else ""
    )

    # Firm prompt — Gemini is told not to use outside knowledge
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context chunks below.
If the answer is not present in the chunks, respond with exactly:
"I don't have enough information to answer that."
Do not use any outside knowledge.
{language_instruction}
Context:
{context}

Question: {question}

Answer:"""

    server_error_waits = [5, 10, 20]

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return response.text

        except errors.ClientError:
            # 429 = free-tier quota exceeded, resets roughly per minute
            print(f"Attempt {attempt} failed (quota exceeded).")
            if attempt == max_retries:
                return "Answer service is temporarily busy. Please try again in a minute."
            print("Waiting 35s for quota to reset...")
            time.sleep(35)

        except errors.ServerError:
            # 503 = model temporarily overloaded on Google's side
            wait = server_error_waits[attempt - 1]
            print(f"Attempt {attempt} failed (server busy). Retrying in {wait}s...")
            if attempt == max_retries:
                return "Gemini is currently overloaded. Please try again in a moment."
            time.sleep(wait)


if __name__ == "__main__":
    # Made-up sample chunks to test with, before real retrieval exists
    sample_chunks = [
        {"text": "The Eiffel Tower is located in Paris, France and was completed in 1889.", "score": 0.91},
        {"text": "Paris is the capital city of France.", "score": 0.85},
        {"text": "The Louvre Museum is one of the largest museums in the world.", "score": 0.62}
    ]

    # Test generate_fast_answer — no API call, instant
    fast = generate_fast_answer(sample_chunks)
    print("Fast answer:", fast)

    # Test generate_polished_answer — question that CAN be answered from chunks
    test_question = "Where is the Eiffel Tower located?"
    polished = generate_polished_answer(test_question, sample_chunks)
    print("\nQuestion:", test_question)
    print("Polished answer:", polished)

    # Test generate_polished_answer — question that CANNOT be answered from chunks
    unrelated_question = "What is the population of Japan?"
    polished2 = generate_polished_answer(unrelated_question, sample_chunks)
    print("\nQuestion:", unrelated_question)
    print("Polished answer:", polished2)

    # Test generate_polished_answer — with a language instruction
    hindi_question = "एफिल टावर कहाँ स्थित है?"
    polished3 = generate_polished_answer(hindi_question, sample_chunks, language_name="Hindi")
    print("\nQuestion:", hindi_question)
    print("Polished answer (Hindi):", polished3)