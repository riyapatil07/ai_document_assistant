# AI-Powered Document Assistant (RAG)

A chatbot that lets you upload a PDF and ask natural-language questions about it. It answers using the actual content of your document instead of guessing, this is what "Retrieval-Augmented Generation" (RAG) means.

**Tech Stack:** Python · LangChain · ChromaDB · Gemini API · Streamlit

---

## 1. How it works (the concept)

A plain LLM only knows what it was trained on, it has never seen your PDF. RAG fixes that by giving the model a "open book exam" instead of a "closed book exam":

```
 PDF Upload
     │
     ▼
 1. Extract text from every page
     │
     ▼
 2. Split text into small overlapping chunks
     │
     ▼
 3. Convert each chunk into a vector (embedding) — a list of numbers
    that represents its MEANING, using Gemini's embedding model
     │
     ▼
 4. Store all those vectors in ChromaDB (a vector database)
     │
     ▼
 User asks a question
     │
     ▼
 5. Convert the question into a vector too, and search ChromaDB for
    the chunks whose vectors are most similar (= most relevant)
     │
     ▼
 6. Send the question + those relevant chunks to Gemini's chat model
     │
     ▼
 7. Gemini generates an answer grounded in your document
```

The key idea: instead of asking the LLM to "remember" your whole PDF, you only ever send it the few small pieces of text that are actually relevant to the current question. That's the "retrieval" part of RAG.

---

## 2. Project structure

```
ai-document-assistant/
├── app.py             # Streamlit UI — chat interface, file upload
├── rag_engine.py       # Core RAG logic — chunking, embeddings, retrieval, generation
├── requirements.txt    # Python dependencies
├── .env.example         # Template for your API key
└── README.md
```

Logic is separated from UI on purpose: `rag_engine.py` has zero Streamlit code in it, so the same RAG pipeline could be reused behind a Django/Flask API later (handy, since you're also building a Django project).

---

## 3. Setup (all free, no credit card needed)

1. **Install Python 3.10+** if you don't already have it.

2. **Create a project folder and add these files** (already done if you got this from Claude).

3. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Get a free Gemini API key:**
   - Go to https://aistudio.google.com/apikey
   - Sign in with a Google account and click "Create API key" no payment info required.

5. **Add your key.** Copy `.env.example` to a new file named `.env` and paste your key in:
   ```
   GOOGLE_API_KEY=your_actual_key_here
   ```

6. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   This opens a browser tab. Upload a PDF, click "Process Document," and start asking questions.

---

