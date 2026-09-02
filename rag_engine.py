

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

    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def build_vectorstore(documents, persist_directory=None, collection_name=None):
    
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

    return "\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )

# ---------------------------------------------------------------------
# STEP 5: The prompt template for generating answers
# ---------------------------------------------------------------------
RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant answering questions about a document.
Use ONLY the context below to answer the question. If the answer isn't
in the context, say you don't know based on the document- do not make
up information.

Context:
{context}

Question:
{question}

Answer clearly and concisely:"""
)


def get_llm():
    
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)


def build_rag_chain(vectorstore):
    
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
    
    answer = rag_chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources
