# HHGOA-SonicQuery — Beginner's Team Guide (3 People)

This guide assumes **zero prior experience** with Git, GitHub, or APIs. Read it top to bottom once as a team, then use it as your daily reference.

**Deadline: August 22, 2026, 11:59 PM**

---

## Part 0: What you're actually building (recap)

A pipeline where someone speaks a question → it gets turned into text → your system finds relevant passages from a dataset → an AI model writes an answer using only those passages.

```
Your voice → Sarvam (speech-to-text) → FAISS (finds relevant chunks) → Gemini (writes the answer)
```

Everything runs inside a **Streamlit** app (a Python tool that turns a script into a website with almost no extra code), hosted for free on **Streamlit Community Cloud**.

---

## Part 1: What each person installs (do this first, individually)

All three of you need:

1. **Python** (3.10 or newer) — download from python.org. During install on Windows, tick "Add Python to PATH."
2. **Git** — download from git-scm.com. This is the tool that lets you save and share code changes.
3. **VS Code** — download from code.visualstudio.com. This is where you'll write code.
4. **A GitHub account** — sign up free at github.com. This is where your code lives online and how you all share it.

Once installed, open a terminal (in VS Code: Terminal → New Terminal) and type these to confirm it worked:
```
python --version
git --version
```
Both should print a version number. If either says "command not found," the install didn't finish — re-run the installer.

---

## Part 2: Git & GitHub explained like you're five

Think of it like a **shared Google Doc, but for code**, with a twist: instead of everyone typing into the same file live, everyone makes their own copy, edits it, and then "merges" their changes back in.

| Term | What it actually means |
|---|---|
| **Repository (repo)** | The project folder, tracked online on GitHub. |
| **Clone** | Download a copy of the repo onto your own laptop. |
| **Commit** | Save a checkpoint of your changes, with a short message describing what you did. |
| **Push** | Upload your commits from your laptop to GitHub. |
| **Pull** | Download everyone else's latest changes from GitHub to your laptop. |
| **Branch** | Your own private working copy inside the repo, so you don't break other people's work while you're mid-edit. |
| **Merge** | Combine your branch's changes back into the main project. |

### One-time setup (whoever creates the repo — pick one person, "Person A")

1. Go to github.com → click **New repository**.
2. Name it something like `HHGOA-SonicQuery`. Keep it **Public**. Click **Create repository**.
3. Add your teammates: go to **Settings → Collaborators → Add people**, type their GitHub usernames, they'll get an email invite. They must accept it.

### Everyone (all 3 people) does this once, after being added as a collaborator:

Open a terminal, navigate to a folder where you want the project (e.g. Desktop), then:

```
git clone https://github.com/YOUR-USERNAME/HHGOA-SonicQuery.git
cd HHGOA-SonicQuery
```

You now have the project on your laptop.

### The daily loop (every time you sit down to work)

**Step 1 — Get the latest version before you start:**
```
git pull
```

**Step 2 — Create your own branch to work in** (do this once per task):
```
git checkout -b your-name-feature
```
Example: `git checkout -b priya-chunking`

**Step 3 — Write your code, save your files as normal in VS Code.**

**Step 4 — Save a checkpoint of your work:**
```
git add .
git commit -m "short description of what I changed"
```

**Step 5 — Upload it to GitHub:**
```
git push origin your-name-feature
```

**Step 6 — Merge it into the main project:**
Go to GitHub.com → your repo → you'll see a yellow banner "Compare & pull request" → click it → click **Create pull request** → then click **Merge pull request**.

That's genuinely the whole workflow. You'll repeat Steps 1–6 many times over the next week.

**If you get a "merge conflict" error:** it means two people edited the same lines of the same file. Don't panic — this is normal. The safest fix as beginners: message each other, agree on whose version to keep, open the conflicted file (Git will mark the conflicting lines with `<<<<<<<` and `>>>>>>>`), delete the version you don't want plus those marker lines, save, then repeat `git add .` → `git commit` → `git push`.

**To avoid conflicts almost entirely:** each person works in a different file (see Part 4 — the split is designed exactly for this).

---

## Part 3: API keys explained from zero

An **API key** is just a password that proves your app is allowed to use someone else's AI service. You paste it into your code, and every time your code "calls" that service, it sends the key along to prove it's you.

**Golden rule: never type an API key directly into a `.py` file that gets pushed to GitHub.** Anyone could see it and use your account. Instead:

1. In your project folder, create a file named exactly `.env`
2. Inside it, write:
```
SARVAM_API_KEY=paste_your_key_here
GEMINI_API_KEY=paste_your_key_here
```
3. Create another file named `.gitignore` (if it doesn't already exist) and add this line inside it:
```
.env
```
This tells Git "never upload this file" — so your keys stay private on your own laptop.

### Getting each key (one person can do this and share the values privately with the team over WhatsApp/Discord — never over GitHub)

**Sarvam (speech-to-text):**
1. Go to sarvam.ai → Sign up.
2. Find the API/Dashboard section → generate an API key.
3. You get free credits automatically — no card needed.

**Gemini (answer generation), via Google AI Studio:**
1. Go to aistudio.google.com → sign in with any Google account.
2. Click **Get API Key** → **Create API key**.
3. Copy it. No card required for the free tier.

Every teammate pastes the same two key values into their **own** `.env` file on their **own** laptop. The `.env` file itself never gets shared through GitHub — only the key values get shared directly, person to person.

---

## Part 4: Splitting the work three ways

Each person owns different files, so you rarely touch the same lines of code — this avoids most merge conflicts.

### Person A — Data & Chunking
**Owns:** `src/chunking.py`
- Load the dataset (`ai4bharat/MSMARCO-XI` from Hugging Face)
- Build 2–3 different chunking strategies (fixed-size, semantic, metadata-aware)
- Output: a saved file of "chunks" ready for the next person

### Person B — Retrieval & Voice Input
**Owns:** `src/retrieval.py` and the voice-input part of `app.py`
- Turn Person A's chunks into embeddings, build the FAISS index
- Wire up the Streamlit microphone input
- Send recorded audio to Sarvam, get back text

### Person C — Generation & Guardrails
**Owns:** `src/generation.py` and `src/guardrails.py`
- Take the retrieved chunks + transcribed question, send to Gemini, get an answer
- Add the "don't answer if not confident" checks
- Add the off-topic/unsafe query filter

**Everyone together, near the end:** connect all three pieces inside `app.py`, test as a team, deploy.

---

## Part 5: The full build plan, step by step

### Step 1 — Set up accounts & keys
Each person: GitHub account (done in Part 2), Sarvam key, Gemini key (Part 3). Only one key set is needed for the whole team, shared privately.

### Step 2 — Create the repo (Part 2, already covered)

### Step 3 — Load and explore the dataset (Person A)
```
pip install datasets
```
```python
from datasets import load_dataset
data = load_dataset("ai4bharat/MSMARCO-XI")
print(data)
print(data["train"][0])
```
Look at what a row actually contains — this decides how chunking should work.

### Step 4 — Build multiple chunking strategies (Person A)
In `src/chunking.py`, write at least:
- **Fixed-size with overlap** — split text into equal-length pieces that slightly overlap so context isn't lost at the edges.
- **Semantic chunking** — split at natural sentence/topic boundaries instead of a fixed character count.
- **Metadata-aware chunking** — keep each original passage/document as its own chunk, using the dataset's existing structure.

Save the output chunks (with their source info attached) to a file, e.g. `chunks.json`, so Person B can load it.

### Step 5 — Build the FAISS index (Person B)
```
pip install sentence-transformers faiss-cpu
```
```python
from sentence_transformers import SentenceTransformer
import faiss

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(list_of_chunks)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "chunks.index")
```

### Step 6 — Voice input + Sarvam STT (Person B)
```
pip install streamlit streamlit-mic-recorder requests
```
Record audio in Streamlit, send it to Sarvam's transcription endpoint, get text back. Wrap the API call in a `try/except` so one failed request doesn't crash the whole app.

### Step 7 — Gemini generation (Person C)
```
pip install google-genai
```
```python
from google import genai
client = genai.Client(api_key="YOUR_KEY")

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents=f"Answer using ONLY this context: {retrieved_chunks}\n\nQuestion: {question}\n\nIf the context doesn't contain the answer, say you don't know."
)
print(response.text)
```

### Step 8 — Guardrails (Person C)
- Before generation: if the top FAISS similarity score is too low, skip Gemini entirely and reply "I don't have enough information to answer that."
- Before that even: a simple keyword/pattern check to catch obviously off-topic or unsafe input.
- After generation: a second, cheap check — ask Gemini "Is this answer supported by this context? yes/no" — and reject the answer if it says no.

### Step 9 — Put it all together in `app.py` (everyone, together, ideally on a call)
```
Streamlit mic input → Sarvam STT → FAISS retrieval → guardrail check → Gemini generation → guardrail check → show answer
```

### Step 10 — Deploy (any one person)
1. Push the finished `app.py` to GitHub (`main` branch).
2. Go to share.streamlit.io → sign in with GitHub → **New app** → pick your repo → Deploy.
3. In the app's **Settings → Secrets**, paste:
```
SARVAM_API_KEY = "your_key"
GEMINI_API_KEY = "your_key"
```
4. Your live link appears at `hhgoa-sonicquery.streamlit.app` — this is your submission link.

### Step 11 — Measure latency
Run 15–20 test queries through your deployed app. For each, log how long chunking+retrieval+generation takes (not counting STT network time). Compute:
- **P50** = the middle value when all times are sorted
- **P70** = the value at the 70th-percentile mark
- **P100** = the slowest single run

A simple way: store each query's time in a Python list, then use `numpy.percentile(times, [50, 70, 100])`.

### Step 12 — Record videos & submit
- 90-second team/process video
- Full demo video
- Post both on Instagram, X, and LinkedIn — **every team member individually posts both videos**, at least one Instagram account public, every post tagged **#RAGInGoa**
- Fill the submission form with your GitHub link and live link

---

## Quick Team Checklist

- [ ] All 3 people have Python, Git, VS Code, GitHub installed/created
- [ ] Repo created, all 3 added as collaborators, all 3 have cloned it
- [ ] Sarvam + Gemini API keys generated, shared privately, added to each person's local `.env`
- [ ] `.env` added to `.gitignore`
- [ ] Person A: chunking done
- [ ] Person B: FAISS + voice input done
- [ ] Person C: generation + guardrails done
- [ ] Everyone merged into `app.py`, tested together
- [ ] Deployed to Streamlit Cloud, secrets added
- [ ] Latency (P50/P70/P100) measured and recorded
- [ ] Both videos recorded
- [ ] Videos posted on IG/X/LinkedIn by all 3 members with #RAGInGoa
- [ ] Submission form filled — **no resubmissions allowed, so double-check before submitting**
