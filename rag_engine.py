"""
rag_engine.py
--------------
This file contains all the "brain" logic of the RAG application.
It is kept separate from app.py (the UI) on purpose -- this is good
practice and something you should mention in interviews: separating
business logic from the UI layer.

WHAT THIS FILE DOES (the RAG pipeline):
1. Reads a PDF and pulls out the raw text.
2. Splits that text into small overlapping "chunks".
3. Converts each chunk into a vector (a list of numbers that represents
   its meaning) using Google's Gemini embedding model.
4. Stores those vectors in ChromaDB, a vector database, so we can
   search them later by meaning instead of by keyword.
5. When the user asks a question, it embeds the question the same way,
   finds the most similar chunks in ChromaDB, and passes those chunks
   + the question to the Gemini chat model to generate a final answer.

This last step (retrieve relevant chunks, then generate an answer using
them) is literally what "Retrieval-Augmented Generation" means.
"""

import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
#from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------------------
# STEP 1: Extract raw text from the uploaded PDF
# ---------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    """
    Takes a file-like object (what Streamlit's file_uploader gives us)
    and returns the full text of the PDF, page by page.
    """
    reader = PdfReader(uploaded_file)
    pages_text = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append({"page": page_number + 1, "text": text})
    return pages_text


# ---------------------------------------------------------------------
# STEP 2: Split the text into chunks
# ---------------------------------------------------------------------
def chunk_text(pages_text, chunk_size=1000, chunk_overlap=150):
    """
    LLMs and embedding models work much better on small chunks of text
    than on a whole document at once. We use a 1000-character chunk size
    with a 150-character overlap so we don't accidentally cut a sentence
    (and its meaning) in half between two chunks.

    Returns a list of LangChain Document objects, which is the standard
    object LangChain uses to represent "a piece of text + metadata".
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    for page in pages_text:
        chunks = splitter.split_text(page["text"])
        for chunk in chunks:
            documents.append(
                Document(page_content=chunk, metadata={"page": page["page"]})
            )
    return documents


# ---------------------------------------------------------------------
# STEP 3 + 4: Embed the chunks and store them in ChromaDB
# ---------------------------------------------------------------------
def get_embeddings_model():
    """
    Returns Google's Gemini embedding model. This turns text into a
    vector of numbers that captures its *meaning*, so two pieces of
    text with similar meaning end up with similar vectors -- even if
    they don't share the same words.
    """
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def build_vectorstore(documents, persist_directory=None, collection_name=None):
    """
    Embeds every document chunk and stores the vectors in FAISS.
    FAISS (Facebook AI Similarity Search) is a fast, reliable vector
    store that runs entirely in memory — no database setup needed.
    """
    import time
    embeddings = get_embeddings_model()

    # Process first batch to create the vectorstore
    batch_size = 5
    all_batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]

    vectorstore = FAISS.from_documents(all_batches[0], embeddings)
    time.sleep(3)

    # Add remaining batches
    for i, batch in enumerate(all_batches[1:], 1):
        vectorstore.add_documents(batch)
        if i < len(all_batches) - 1:
            time.sleep(3)

    return vectorstore


def format_docs(docs):
    """Joins retrieved chunks into a single text block for the prompt,
    and labels each one with its source page so answers can be traced
    back to the document."""
    return "\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )

# ---------------------------------------------------------------------
# STEP 5: The prompt template for generating answers
# ---------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about a document.
Use ONLY the context below to answer the question. If the answer isn't
in the context, say you don't know based on the document -- do not make
up information.

Context:
{context}

Question:
{question}

Answer clearly and concisely:"""
)


def get_llm():
    """
    Returns the Gemini chat model that generates the final answer.
    gemini-2.5-flash is fast, free-tier friendly, and great for RAG.
    """
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


def build_rag_chain(vectorstore):
    """
    Builds the full RAG chain using LCEL (LangChain Expression Language),
    the modern way to compose LangChain pipelines with the `|` operator.

    The chain does this, in order:
      1. retriever -> finds the top-k most relevant chunks for the question
      2. format_docs -> turns those chunks into one text block
      3. RAG_PROMPT -> inserts {context} and {question} into the prompt
      4. llm -> Gemini generates the answer
      5. StrOutputParser -> extracts plain text from the model's response
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


def answer_question(rag_chain, retriever, question):
    """
    Runs the chain to get an answer, and separately fetches the source
    chunks so the UI can show "here's where this answer came from" --
    a nice touch that makes the app feel trustworthy.
    """
    answer = rag_chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources
