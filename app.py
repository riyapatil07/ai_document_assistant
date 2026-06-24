"""
app.py
-------
This is the Streamlit web interface. Streamlit turns a plain Python
script into a web app -- every time the user interacts with something
(uploads a file, types a message), this whole script re-runs from top
to bottom, and Streamlit only redraws what changed. That's why we use
`st.session_state` to remember things (like the vector store and chat
history) between those re-runs.

Run this with:  streamlit run app.py
"""

import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

import rag_engine

load_dotenv()  # loads GOOGLE_API_KEY from a local .env file, if present

st.set_page_config(page_title="AI Document Assistant", page_icon="📄")
st.title("📄 AI-Powered Document Assistant")
st.caption("Upload a PDF and ask questions about it — powered by RAG (LangChain + ChromaDB + Gemini)")

# -----------------------------------------------------------------
# Session state: things that need to survive between re-runs
# -----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []          # chat history: list of {"role", "content"}
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False

# -----------------------------------------------------------------
# Sidebar: API key + PDF upload
# -----------------------------------------------------------------
with st.sidebar:
    st.header("Setup")

    api_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    process_clicked = st.button(
        "Process Document",
        disabled=not (uploaded_file and os.getenv("GOOGLE_API_KEY")),
        use_container_width=True,
    )

    if not os.getenv("GOOGLE_API_KEY"):
        st.info("Enter your free Gemini API key to get started. "
                "Get one at aistudio.google.com/apikey")

# -----------------------------------------------------------------
# Process the uploaded PDF into a vector store + RAG chain
# -----------------------------------------------------------------
if process_clicked and uploaded_file:
    with st.spinner("Reading PDF, creating chunks, and building embeddings..."):
        try:
            pages_text = rag_engine.extract_text_from_pdf(uploaded_file)
            documents = rag_engine.chunk_text(pages_text)

            if not documents:
                st.error("Couldn't extract any text from this PDF. "
                          "It may be a scanned/image-only PDF.")
            else:
                st.info(f"PDF read successfully. {len(documents)} chunks created. Now building embeddings...")

                # Each session gets its own throwaway Chroma collection
                persist_dir = tempfile.mkdtemp(prefix="chroma_")
                vectorstore = rag_engine.build_vectorstore(documents, persist_dir)
                rag_chain, retriever = rag_engine.build_rag_chain(vectorstore)

                st.session_state.rag_chain = rag_chain
                st.session_state.retriever = retriever
                st.session_state.doc_processed = True
                st.session_state.messages = []  # reset chat for the new document

                st.success(f"Document processed into {len(documents)} chunks. Ask away!")
        except Exception as e:
            # st.error(f"Something went wrong while processing the PDF: {e}")
            st.error(f"EXACT ERROR: {e}")
            import traceback
            st.code(traceback.format_exc())

# -----------------------------------------------------------------
# Chat interface
# -----------------------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.doc_processed:
    st.info("Upload a PDF and click **Process Document** in the sidebar to start chatting.")
else:
    user_question = st.chat_input("Ask a question about your document...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = rag_engine.answer_question(
                        st.session_state.rag_chain,
                        st.session_state.retriever,
                        user_question,
                    )
                    st.markdown(answer)

                    with st.expander("Sources used for this answer"):
                        for doc in sources:
                            st.markdown(f"**Page {doc.metadata.get('page', '?')}:** {doc.page_content[:200]}...")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating an answer: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
