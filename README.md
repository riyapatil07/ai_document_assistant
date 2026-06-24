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

## 4. Code walkthrough (map this to your resume bullets)

| Resume bullet | Where it lives in the code |
|---|---|
| "Built a RAG application enabling users to upload PDF documents and ask natural-language questions" | `app.py` file uploader + chat input |
| "Implemented document chunking, embeddings, and semantic search using LangChain and ChromaDB" | `rag_engine.py` — `chunk_text()`, `build_vectorstore()` |
| "Integrated Gemini API to generate context-aware answers" | `rag_engine.py` — `get_llm()`, `build_rag_chain()` |
| "Developed a conversational interface" | `app.py` — `st.chat_message` / `st.chat_input` + `st.session_state.messages` |

---

## 5. Likely interview questions for this project (and how to answer)

**Q: What is RAG and why use it instead of fine-tuning?**
A: RAG retrieves relevant info at query time and feeds it to the model, instead of retraining the model on your data. It's cheaper, instant to update (just re-index a new PDF, no retraining), and avoids the model "hallucinating" since answers are grounded in retrieved text.

**Q: What are embeddings?**
A: Embeddings are vectors (lists of numbers) that represent the meaning of text. Pieces of text with similar meaning end up close together in that vector space, so you can find relevant content by measuring vector distance instead of matching exact keywords.

**Q: Why did you chunk the document instead of sending the whole PDF to the model?**
A: Two reasons — embedding models work better on focused, smaller pieces of text, and LLMs have a limited context window plus perform better when given only the relevant information rather than an entire document.

**Q: How did you pick chunk size and overlap?**
A: I used ~1000 characters per chunk with 150 characters of overlap, so a sentence isn't awkwardly cut between two chunks and important context near a chunk boundary isn't lost.

**Q: What is ChromaDB and why a vector database instead of a normal SQL database?**
A: ChromaDB stores embeddings and is optimized for similarity search (finding "nearest" vectors), which a normal relational database isn't built for.

**Q: What would you improve if you had more time?**
A: Good honest answers — see the section below.

---

## 6. Possible future improvements

These are good things to mention if asked "what would you add next":

- Support multiple PDFs at once, and let the user pick which document(s) to search.
- Add conversation memory so follow-up questions ("what about page 3?") use the chat history, not just the latest message.
- Show a confidence/relevance score next to retrieved sources.
- Swap ChromaDB for a hosted vector DB (e.g., Pinecone) for a production deployment.
- Add evaluation: a small set of test questions with expected answers, to measure retrieval accuracy.
- Deploy it (Streamlit Community Cloud is free) so you can link a live demo on your resume, not just GitHub code.

---

## 7. Free tier note

Gemini's free tier (via Google AI Studio) covers `gemini-2.5-flash` for chat generation and `gemini-embedding-001` for embeddings, with no credit card required — generous enough for a portfolio project. If you ever see a 429 error, you've briefly hit the per-minute rate limit; just wait a bit and try again.
