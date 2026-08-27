import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Sync environment variables
load_dotenv()

# Import backend business logic modules
from indexer import index_repository
from query_engine import search_chunks, generate_answer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CodeRAG-API")

app = FastAPI(
    title="CodeRAG API Backend",
    description="REST API service for GitHub codebase vector indexing & RAG natural language Q&A",
    version="1.0.0"
)

# Enable CORS for frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (Vite dev server, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Data Models
class IndexRequest(BaseModel):
    repo_url: str = Field(..., example="https://github.com/owner/repo", description="Public GitHub repository URL")

class IndexResponse(BaseModel):
    success: bool
    repo_url: str
    files_processed: int
    chunks_created: int
    message: str

class QueryRequest(BaseModel):
    repo_url: str = Field(..., example="https://github.com/owner/repo")
    question: str = Field(..., example="How is routing implemented in this project?")
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

class CodeChunkSource(BaseModel):
    id: Optional[int] = None
    file_path: str
    start_line: int
    end_line: int
    chunk_text: str
    language: str
    similarity: float
    distance: float

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    sources: List[CodeChunkSource]


@app.get("/api/health")
def health_check():
    """Health check endpoint confirming API server readiness and configuration."""
    db_configured = bool(os.getenv("DATABASE_URL"))
    gemini_configured = bool(os.getenv("GEMINI_API_KEY")) and os.getenv("GEMINI_API_KEY") != "your_gemini_api_key_here"
    
    return {
        "status": "online",
        "database_configured": db_configured,
        "gemini_api_configured": gemini_configured,
        "service": "CodeRAG Backend API"
    }


@app.post("/api/index", response_model=IndexResponse)
def handle_index_repository(request: IndexRequest):
    """
    Clones GitHub repository, chunks source files, computes embeddings, and updates pgvector database.
    """
    clean_url = request.repo_url.strip()
    if not clean_url:
        raise HTTPException(status_code=400, detail="Repository URL cannot be empty.")
    if not (clean_url.startswith("https://github.com/") or clean_url.startswith("http://github.com/")):
        raise HTTPException(status_code=400, detail="Invalid repository URL. Must start with https://github.com/")

    logger.info(f"Received index request for: {clean_url}")
    try:
        summary = index_repository(clean_url)
        
        message = (
            f"Successfully indexed {summary['files_processed']} files into {summary['chunks_created']} vector chunks."
            if summary["chunks_created"] > 0
            else "Repository cloned, but no supported source files (.py, .js, .ts, .java, .c, .cpp) were found."
        )
        
        return IndexResponse(
            success=True,
            repo_url=summary["repo_url"],
            files_processed=summary["files_processed"],
            chunks_created=summary["chunks_created"],
            message=message
        )
    except Exception as e:
        logger.error(f"Error indexing repository {clean_url}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index repository: {str(e)}"
        )


@app.post("/api/query", response_model=QueryResponse)
def handle_query_codebase(request: QueryRequest):
    """
    Retrieves top_k matching code chunks for user question and generates an answer using Google Gemini.
    """
    clean_repo = request.repo_url.strip()
    clean_question = request.question.strip()

    if not clean_repo:
        raise HTTPException(status_code=400, detail="Repository URL is required.")
    if not clean_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    logger.info(f"Received question query for repo: {clean_repo} - Question: '{clean_question}'")
    try:
        # Step 1: Perform vector search
        retrieved_chunks = search_chunks(
            question=clean_question,
            repo_url=clean_repo,
            top_k=request.top_k or 5
        )

        # Step 2: Generate RAG answer
        answer_text = generate_answer(
            question=clean_question,
            retrieved_chunks=retrieved_chunks
        )

        sources = [
            CodeChunkSource(
                id=c.get("id"),
                file_path=c["file_path"],
                start_line=c["start_line"],
                end_line=c["end_line"],
                chunk_text=c["chunk_text"],
                language=c["language"],
                similarity=c["similarity"],
                distance=c["distance"]
            )
            for c in retrieved_chunks
        ]

        return QueryResponse(
            success=True,
            question=clean_question,
            answer=answer_text,
            sources=sources
        )
    except Exception as e:
        logger.error(f"Error querying repository {clean_repo}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error answering question: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
