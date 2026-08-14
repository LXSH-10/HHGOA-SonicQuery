# HHGOA-SonicQuery — Day-by-Day Master Plan

**Today: Friday, Aug 14, 2026 → Deadline: Saturday, Aug 22, 11:59 PM**
That's 9 days. This plan uses all of them, with a buffer built in near the end.

Read this alongside the earlier **Beginner's Team Guide** (that one explains git/GitHub/API keys from zero — keep it open in another tab). This document tells you **exactly what to do, which day, using which tool.**

**Team roles (unchanged):**
- **Person A** → Data & Chunking
- **Person B** → Retrieval (FAISS) & Voice Input (Sarvam)
- **Person C** → Answer Generation (Gemini) & Guardrails

---

## The Toolbox — what each tool is, when you'll use it, how to use it

Skim this once now, then come back to the relevant section on the day you need it.

### 1. Git & GitHub
- **What:** how your team shares code without overwriting each other.
- **When:** every single day, before and after you code.
- **How:** `git pull` → `git checkout -b your-branch` → code → `git add .` → `git commit -m "message"` → `git push origin your-branch` → open a Pull Request on GitHub → Merge. (Full explanation already in the Beginner's Team Guide.)

### 2. Hugging Face `datasets` library
- **What:** downloads the MSMARCO-XI dataset into Python.
- **When:** Day 2, Person A.
- **How:**
```
pip install datasets
```
```python
from datasets import load_dataset
data = load_dataset("ai4bharat/MSMARCO-XI")
print(data["train"][0])
```

### 3. sentence-transformers
- **What:** turns text into a list of numbers ("embeddings") so a computer can measure how *similar* two pieces of text are.
- **When:** Day 2–3, Person A (for semantic chunking) and Person B (for building the search index).
- **How:**
```
pip install sentence-transformers
```
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
vectors = model.encode(["some text here", "more text"])
```

### 4. FAISS
- **What:** a super-fast search engine for those number-vectors — given a question's vector, it instantly finds the most similar chunks.
- **When:** Day 3, Person B.
- **How:**
```
pip install faiss-cpu
```
```python
import faiss
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
faiss.write_index(index, "chunks.index")   # save
# later: index = faiss.read_index("chunks.index")
```

### 5. Sarvam (speech-to-text)
- **What:** converts recorded voice into text.
- **When:** Day 4, Person B.
- **How:** their current recommended model is `saaras:v3` with `mode="transcribe"` (the older `saarika` model is being phased out). Works best with 16kHz audio.
```
pip install requests
```
```python
import requests

url = "https://api.sarvam.ai/speech-to-text"
headers = {"api-subscription-key": "YOUR_SARVAM_KEY"}
files = {"file": open("question.wav", "rb")}
data = {"model": "saaras:v3", "mode": "transcribe"}

response = requests.post(url, headers=headers, files=files, data=data)
transcript = response.json()["transcript"]
```
Sarvam also has an official Python SDK if you'd rather not hand-roll the request — check their docs' "Libraries & SDKs" page for the install command, both work.

### 6. Google GenAI SDK (Gemini, via Google AI Studio)
- **What:** the AI model that writes the final answer from retrieved chunks.
- **When:** Day 4–5, Person C.
- **How:**
```
pip install google-genai
```
```python
from google import genai
client = genai.Client(api_key="YOUR_GEMINI_KEY")

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents=f"Answer using ONLY this context: {chunks}\n\nQuestion: {question}\n\nIf the answer isn't in the context, say you don't know."
)
print(response.text)
```

### 7. Streamlit + streamlit-mic-recorder
- **What:** turns your Python script into an actual website with a working microphone button.
- **When:** Day 4 onward (Person B starts it), full integration Day 6 (everyone).
- **How:**
```
pip install streamlit streamlit-mic-recorder
```
```python
import streamlit as st
from streamlit_mic_recorder import mic_recorder

st.title("HHGOA-SonicQuery")
audio = mic_recorder(start_prompt="Ask a question", stop_prompt="Stop")
if audio:
    # audio["bytes"] contains the recorded audio — send this to Sarvam
    pass
```
Run locally with: `streamlit run app.py`

### 8. numpy (for latency numbers)
- **What:** calculates your P50/P70/P100 latency stats.
- **When:** Day 7.
- **How:**
```python
import numpy as np
times = [0.12, 0.18, 0.09, ...]  # seconds, one per test query
p50, p70, p100 = np.percentile(times, [50, 70, 100])
```

---

## Day-by-Day Plan

### Day 1 — Fri Aug 14 (today): Setup Day
**Everyone, together, ideally on a call:**
- [ ] Install Python, Git, VS Code (Part 1 of Beginner Guide)
- [ ] Create the GitHub repo, add both teammates as collaborators, everyone clones it
- [ ] Sign up for Sarvam + Google AI Studio, generate both API keys, share them privately (not on GitHub), each person creates their own local `.env`
- [ ] Confirm roles (A/B/C) and re-read the Toolbox section above once as a group
- [ ] Create the empty project structure in the repo and push it:
```
src/chunking.py
src/retrieval.py
src/generation.py
src/guardrails.py
app.py
requirements.txt
.env          (not pushed — in .gitignore)
.gitignore
```

### Day 2 — Sat Aug 15: Foundations
- **Person A:** load the dataset, explore 20–30 sample rows, note language(s) and structure. Start writing `chunking.py` — get the fixed-size-with-overlap strategy working first (simplest one).
- **Person B:** read Sarvam's docs, get one test transcription working from a sample audio file (doesn't need to be wired into anything yet — just prove the API call works).
- **Person C:** get one test Gemini call working (just send it a plain question, no retrieval yet — prove the API key and call work).

### Day 3 — Sun Aug 16: Chunking + Embeddings
- **Person A:** finish semantic chunking and metadata-aware chunking strategies. Save all chunks (with source metadata) to `chunks.json`. Push to GitHub.
- **Person B:** `git pull` to get Person A's `chunks.json`. Embed the chunks with sentence-transformers, build the FAISS index, save it as `chunks.index`. Push to GitHub.
- **Person C:** design the actual prompt template for Gemini (the instruction that tells it to only answer from context, and say "I don't know" otherwise). Test it manually with a few fake retrieved chunks.

### Day 4 — Mon Aug 17: Voice Input + Wiring Generation
- **Person A:** free day / help wherever needed (or start writing the README project description, or start drafting the process-video script).
- **Person B:** `pull` latest. Build the Streamlit mic input, wire it to Sarvam, confirm recorded voice → correct text end to end.
- **Person C:** `pull` latest FAISS index. Write `retrieval.py`'s counterpart on the generation side: take a real question, get top-k chunks from FAISS, feed them into the Gemini prompt, confirm it returns a real grounded answer.

### Day 5 — Tue Aug 18: Guardrails
- **Person C (lead):** add the two guardrail checks:
  1. Before generation — if FAISS's top similarity score is below a threshold, skip Gemini and return "I don't have enough information to answer that."
  2. After generation — a second cheap Gemini call: "Is this answer supported by this context? yes/no" — reject if "no."
  Also add a simple off-topic/unsafe query filter before Sarvam even runs.
- **Person A & B:** test each other's pieces individually (Person B tests Person A's chunking output quality; Person A tests Person B's retrieval results make sense). Fix any bugs found.

### Day 6 — Wed Aug 19: Full Integration Day (everyone, same call/room if possible)
- Wire everything into `app.py` in order: **mic input → Sarvam → FAISS retrieval → guardrail check → Gemini → guardrail check → display answer.**
- This is the day most likely to hit "it doesn't work when combined" bugs — budget the whole day for it, don't rush.
- Once it runs locally for all three of you, push the final `app.py` to `main`.

### Day 7 — Thu Aug 20: Deploy + Latency Testing
- **Any one person:** deploy to Streamlit Community Cloud (share.streamlit.io → New app → add API keys under Settings → Secrets).
- **Everyone:** run 15–20 varied test questions through the live app. Log the time for chunking+retrieval+generation on each (excluding the STT network call). Compute P50/P70/P100 with numpy.
- Fix anything that broke specifically in the deployed version (deployed environments sometimes behave differently than local).

### Day 8 — Fri Aug 21: Videos + Buffer
- Record the 90-second team/process video and the full demo video.
- Every team member posts both videos individually on Instagram, X, and LinkedIn, tagged **#RAGInGoa**. Confirm at least one Instagram account is public.
- Use any spare time today as buffer for bugs, not new features — this close to the deadline, stability beats extra polish.

### Day 9 — Sat Aug 22: Submit
- Final check: live link works, GitHub repo is public and pushed, both videos are posted by all three members with the correct hashtag.
- Fill out the submission form early in the day — **no resubmissions are allowed**, so don't leave this until 11:58 PM.

---

## Daily 10-Minute Standup (do this every morning, even just over chat)
1. What did I finish yesterday?
2. What am I doing today?
3. Is anything blocking me (waiting on a teammate's file, a bug, an API not working)?

This alone prevents most of the "wait, I didn't know you needed that" problems in a first-time team project.
