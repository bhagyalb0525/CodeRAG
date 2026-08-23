import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables (.env locally, or st.secrets on Streamlit Cloud)
load_dotenv()
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ[k] = str(v)
except Exception:
    pass

# Import Phase 1 and Phase 2 backend modules
from indexer import index_repository
from query_engine import search_chunks, generate_answer

# Streamlit Page Config
st.set_page_config(
    page_title="Chat with your Codebase RAG",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State
if "indexed_repo_url" not in st.session_state:
    st.session_state.indexed_repo_url = ""
if "files_processed" not in st.session_state:
    st.session_state.files_processed = 0
if "chunks_created" not in st.session_state:
    st.session_state.chunks_created = 0
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Sidebar: Explainer & Info
with st.sidebar:
    st.title("⚙️ CodeRAG System")
    st.markdown("### How this works")
    st.info(
        "This tool clones your GitHub repo, breaks the source code into chunks, "
        "embeds them locally using `sentence-transformers`, and uses RAG to answer "
        "your questions using only your actual code as context."
    )
    
    st.divider()
    
    # Status panel
    if st.session_state.indexed_repo_url:
        st.success("🟢 Active Repository Indexed")
        st.markdown(f"**Repo:** `{st.session_state.indexed_repo_url}`")
        col_s1, col_s2 = st.columns(2)
        col_s1.metric("Files", st.session_state.files_processed)
        col_s2.metric("Chunks", st.session_state.chunks_created)
    else:
        st.warning("⚪ No repository indexed yet.")

    st.divider()
    st.caption("Built with Streamlit, PostgreSQL + pgvector, sentence-transformers & Gemini API.")


# Main UI Header
st.title("💻 Chat with your Codebase")
st.markdown("Point to any public GitHub repository, index its source code, and ask natural language questions.")

st.divider()

# Section 1: Repository Ingestion & Indexing
st.subheader("1. Index a GitHub Repository")

col_input, col_btn = st.columns([4, 1])

with col_input:
    repo_url_input = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repository",
        value=st.session_state.indexed_repo_url,
        help="Enter a public GitHub repository link containing source code files (.py, .js, .ts, .java, .c, .cpp)"
    )

with col_btn:
    st.write(" ") # Vertical alignment spacer
    index_clicked = st.button("Index Repository", type="primary", use_container_width=True)

if index_clicked:
    clean_url = repo_url_input.strip()
    if not clean_url:
        st.error("Please enter a valid GitHub repository URL.")
    elif not (clean_url.startswith("https://github.com/") or clean_url.startswith("http://github.com/")):
        st.error("URL must start with `https://github.com/`")
    else:
        with st.spinner(f"Cloning & indexing repository... (This may take a minute for embeddings)"):
            try:
                summary = index_repository(clean_url)
                
                # Update session state to persist indexed repo metadata
                st.session_state.indexed_repo_url = summary["repo_url"]
                st.session_state.files_processed = summary["files_processed"]
                st.session_state.chunks_created = summary["chunks_created"]
                st.session_state.qa_history = [] # Reset Q&A history on new repo index
                
                if summary["chunks_created"] > 0:
                    st.success(
                        f" Successfully indexed **{summary['files_processed']}** files "
                        f"into **{summary['chunks_created']}** vector chunks!"
                    )
                else:
                    st.warning(
                        "Repository cloned, but no supported source files (.py, .js, .ts, .java, .c, .cpp) were found."
                    )
            except Exception as err:
                st.error(f"Failed to index repository: {err}")

st.divider()

# Section 2: Q&A Query Interface
st.subheader("2. Ask Natural Language Questions")

if not st.session_state.indexed_repo_url or st.session_state.chunks_created == 0:
    st.info("👈 Index a repository above to enable questions.")
else:
    st.caption(f"Currently chatting with: `{st.session_state.indexed_repo_url}`")
    
    question_input = st.text_input(
        "Enter your question about the codebase:",
        placeholder="e.g. How is routing handled? Where is authentication defined?",
        key="user_question_field"
    )
    
    col_q1, col_q2 = st.columns([1, 4])
    with col_q1:
        ask_clicked = st.button("Ask Question", type="primary", use_container_width=True)
    
    if ask_clicked:
        clean_q = question_input.strip()
        if not clean_q:
            st.warning("Please enter a question before clicking Ask.")
        else:
            with st.spinner("Searching code chunks and generating answer with Gemini..."):
                try:
                    # Retrieve top 5 most similar vector chunks
                    retrieved_chunks = search_chunks(
                        question=clean_q,
                        repo_url=st.session_state.indexed_repo_url,
                        top_k=5
                    )
                    
                    # Generate RAG answer
                    answer_text = generate_answer(
                        question=clean_q,
                        retrieved_chunks=retrieved_chunks
                    )
                    
                    # Append to session state history
                    st.session_state.qa_history.insert(0, {
                        "question": clean_q,
                        "answer": answer_text,
                        "sources": retrieved_chunks
                    })
                except Exception as q_err:
                    st.error(f"Error answering question: {q_err}")

# Display Q&A History
if st.session_state.qa_history:
    st.markdown("### Answer")
    
    for idx, item in enumerate(st.session_state.qa_history):
        with st.container(border=True):
            st.markdown(f"#### ❓ Question: {item['question']}")
            st.markdown(item["answer"])
            
            # Sources section (Required build requirement)
            with st.expander(f"📁 Sources used ({len(item['sources'])} code chunks)"):
                if not item["sources"]:
                    st.write("No matching code chunks were retrieved.")
                for s_idx, src in enumerate(item["sources"], start=1):
                    st.markdown(
                        f"**Source #{s_idx}:** `{src['file_path']}` (Lines {src['start_line']}-{src['end_line']}) "
                        f"| *Similarity score: {src['similarity']}*"
                    )
                    st.code(src["chunk_text"], language=src["language"])
                    if s_idx < len(item["sources"]):
                        st.divider()
