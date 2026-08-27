# CodeRAG - Chat with your Codebase RAG Application

A full-stack RAG (Retrieval-Augmented Generation) application allowing users to point to any public GitHub repository and ask natural language questions about its source code.

Built with **FastAPI**, **React (Vite + Lucide)**, **PostgreSQL + pgvector**, **sentence-transformers**, and **Google Gemini API**.

---

## 🏗️ Architecture & Stack

- **React Frontend (`frontend/`)**: Modern dark glassmorphic UI built with Vite, React, Lucide Icons, and React Markdown.
- **FastAPI Backend Server (`api.py`)**: REST API endpoints for repository ingestion (`/api/index`), vector Q&A search (`/api/query`), and health monitoring (`/api/health`).
- **`indexer.py`**: Git clone ingestion, source code chunking (50-line blocks), local 384d vector embedding (`sentence-transformers/all-MiniLM-L6-v2`), and `pgvector` database storage.
- **`query_engine.py`**: Cosine similarity vector search (`<=>` operator) and grounded RAG answer generation using **Google Gemini API**.

---

## 💻 Local Development

### 1. Prerequisites
- Python 3.11+
- Node.js 18+
- PostgreSQL database with `pgvector` extension enabled (or Supabase / Neon / Docker container)
- Google Gemini API Key ([Get a free key here](https://aistudio.google.com/app/apikey))

### 2. Set Up Environment Variables (`.env`)
Create a `.env` file in the root folder:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/coderag
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run Backend API
```bash
# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI server (runs on http://localhost:8000)
python -m uvicorn api:app --reload --port 8000
```

### 4. Run React Frontend
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 🐳 Option A: All-in-One Docker Deployment (Recommended for Local/VPS)

You can run the entire stack (PostgreSQL + pgvector, FastAPI, React + Nginx) with a single command using Docker Compose:

```bash
# 1. Set GEMINI_API_KEY in your environment or .env file
export GEMINI_API_KEY="your_gemini_api_key_here"

# 2. Build and start all services
docker-compose up --build -d
```

- **React Frontend**: `http://localhost:3000`
- **FastAPI Backend**: `http://localhost:8000`
- **PostgreSQL pgvector DB**: `localhost:5432`

To stop services:
```bash
docker-compose down
```

---

## ☁️ Option B: Free Cloud Deployment (Vercel + Render + Neon/Supabase)

### Step 1: Managed Cloud Database (Neon / Supabase)
1. Create a free PostgreSQL database on [Neon.tech](https://neon.tech) or [Supabase.com](https://supabase.com).
2. Ensure `pgvector` extension is enabled (handled automatically by `indexer.py`).
3. Copy your `DATABASE_URL` connection string.

### Step 2: Deploy Backend API (Render / Railway / Koyeb)
1. Push your repository to GitHub.
2. Log in to [Render.com](https://render.com) and create a **Web Service**.
3. Connect your repository.
4. Set settings:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Add Environment Variables:
   - `DATABASE_URL`: Your cloud PostgreSQL URI
   - `GEMINI_API_KEY`: Your Gemini API key
6. Copy your deployed backend service URL (e.g., `https://coderag-api.onrender.com`).

### Step 3: Deploy React Frontend (Vercel / Netlify)
1. Log in to [Vercel.com](https://vercel.com) and click **Add New Project**.
2. Select your GitHub repository.
3. Configure project settings:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
4. Click **Deploy**. Vercel will build and publish your React app with instant HTTPS!

---

## 📂 Project Structure

```
.
├── api.py                  # FastAPI REST API server
├── indexer.py              # Repository scanner, chunker & vector storage
├── query_engine.py         # pgvector similarity search & Gemini RAG
├── requirements.txt        # Backend dependencies
├── Dockerfile.backend      # Dockerfile for FastAPI
├── Dockerfile.frontend     # Dockerfile for React (Nginx)
├── docker-compose.yml      # Orchestration for DB + API + Frontend
├── nginx.conf              # Nginx server configuration for React app
└── frontend/               # React application (Vite)
    ├── src/
    │   ├── components/     # Header, MetricsBar, RepoIndexer, ChatInterface, SourceInspector
    │   ├── App.jsx         # Main React App component
    │   └── index.css       # Dark glassmorphic design system
    ├── vercel.json         # Vercel SPA routing rules
    └── package.json
```
