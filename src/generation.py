from dotenv import load_dotenv
import os
import time
from google import genai
from google.genai import errors

load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=gemini_key)


def generate_answer(question, retrieved_chunks, max_retries=3):
    """
    Calls Gemini and instructs it to answer ONLY using the given chunks.
    If the answer isn't present in them, it returns the exact fallback
    message instead of guessing. Retries automatically if Gemini's
    servers are temporarily overloaded (503 error).
    """
    # Join all chunk texts into one labeled context block
    context = "\n\n".join(
        f"Chunk {i+1}: {chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    )

    # Firm prompt — Gemini is told not to use outside knowledge
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context chunks below.
If the answer is not present in the chunks, respond with exactly:
"I don't have enough information to answer that."
Do not use any outside knowledge.

Context:
{context}

Question: {question}

Answer:"""

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt
            )
            return response.text
        except errors.ServerError as e:
            # 503 = model temporarily overloaded on Google's side
            print(f"Attempt {attempt} failed (server busy). Retrying in 5s...")
            if attempt == max_retries:
                return "Gemini is currently overloaded. Please try again in a moment."
            time.sleep(5)


if __name__ == "__main__":
    # Made-up sample chunks to test with, before real retrieval exists
    sample_chunks = [
        {"text": "The Eiffel Tower is located in Paris, France and was completed in 1889."},
        {"text": "Paris is the capital city of France."},
        {"text": "The Louvre Museum is one of the largest museums in the world."}
    ]

    # Test 1: question that CAN be answered from the chunks
    test_question = "Where is the Eiffel Tower located?"
    answer = generate_answer(test_question, sample_chunks)
    print("Question:", test_question)
    print("Answer:", answer)

    # Test 2: question that CANNOT be answered from the chunks
    unrelated_question = "What is the population of Japan?"
    answer2 = generate_answer(unrelated_question, sample_chunks)
    print("\nQuestion:", unrelated_question)
    print("Answer:", answer2)