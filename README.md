# CodeRAG - Chat with your Codebase RAG Application

A full-stack RAG (Retrieval-Augmented Generation) application allowing users to point to any GitHub repository and ask natural language questions about its source code.

Built with **Python 3.11+**, **Streamlit**, **PostgreSQL + pgvector**, **sentence-transformers**, and **Google Gemini API**.

---

## 🏗️ Architecture & Modules

1. **`indexer.py` (Phase 1 Backend)**:
   - Ingests GitHub repository code using GitPython / git clone.
   - Filters source code files (`.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`), capping at 500 smallest files.
   - Chunks files into 50-line blocks with 10-line overlaps.
   - Generates 384-dim embeddings locally via `sentence-transformers/all-MiniLM-L6-v2`.
   - Stores chunks & vectors in PostgreSQL equipped with the `pgvector` extension.

2. **`query_engine.py` (Phase 2 Query Engine)**:
   - Embeds user questions using `all-MiniLM-L6-v2`.
   - Performs cosine similarity search (`<=>` distance operator) in `pgvector` to fetch the top $k$ relevant code chunks.
   - Constructs grounded prompts with file path & line range metadata.
   - Generates precise answers using **Google Gemini API**.

3. **`app.py` (Phase 2 Streamlit UI)**:
   - Web application providing repository indexing and an interactive Q&A interface.
   - Displays clean answers along with an expandable **Sources** viewer highlighting used code chunks, line numbers, and file paths.
   - Persists repo indexing status in session state.

---

## 🚀 Quick Setup Guide

### 1. Prerequisites
- Python 3.11+
- Git
- PostgreSQL database with `pgvector` (Supabase recommended or Docker container)
- Google Gemini API Key ([Get a free key here](https://aistudio.google.com/app/apikey))

---

### 2. Installation

```bash
# Clone or open workspace
cd d:\CodeRAG

# Create & activate virtual environment (optional)
python -m venv venv
# Windows (PowerShell): .\venv\Scripts\Activate.ps1
# macOS/Linux: source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

### 3. Environment Variables (`.env`)

Create a `.env` file in the root folder (or copy `.env.example`):

```bash
cp .env.example .env
```

Set your configuration in `.env`:
```env
# PostgreSQL connection string (Supabase URI recommended)
DATABASE_URL=postgresql://postgres.yourprojectref:yourpassword@aws-0-us-east-1.pooler.supabase.com:5432/postgres

# Google Gemini API key
GEMINI_API_KEY=AIzaSy...your_gemini_api_key_here
```

---

## 💻 Running the Application

### 🎈 Option A: Launch Streamlit Web UI (Recommended)

Start the interactive Streamlit application:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser:
1. Paste your GitHub repository URL (e.g. `https://github.com/bottlepy/bottle`).
2. Click **Index Repository**.
3. Type your question (e.g. *"How does request routing work in bottle?"*) and click **Ask Question**.
4. Expand **Sources** to inspect exact code snippets and line numbers used for the answer.

---

### 🖥️ Option B: Run Query Engine via CLI

You can also run search & RAG query generation directly from the command line:

```bash
python query_engine.py https://github.com/bottlepy/bottle "How is routing implemented?"
```

---

## 📂 Workspace Structure

```
.
├── app.py              # Streamlit frontend UI
├── indexer.py          # Phase 1 backend ingestion, chunking, embedding, pgvector storage
├── query_engine.py     # Phase 2 vector search & Gemini RAG answer generation
├── requirements.txt    # Project dependencies
├── .env.example        # Environment variable template
└── README.md           # Documentation & instructions
```
