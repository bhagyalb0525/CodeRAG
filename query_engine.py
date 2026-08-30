import os
import sys
import logging
import warnings
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Third-party imports
try:
    import psycopg2
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector
except ImportError:
    psycopg2 = None
    register_vector = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("QueryEngine")


def sync_secrets():
    load_dotenv()
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            sec = st.secrets
            for k in list(sec.keys()):
                try:
                    val = sec[k]
                    if isinstance(val, str):
                        os.environ[k] = val
                except Exception:
                    pass
    except Exception:
        pass


def search_chunks(
    question: str,
    repo_url: str,
    top_k: int = 5,
    db_url: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    1. Embeds user question using Google Gemini Embedding API (gemini-embedding-001).
    2. Executes cosine similarity search (<=> operator) against pgvector code_chunks table.
    3. Returns top_k matching chunks with file path, line numbers, text, language, and similarity score.
    """
    sync_secrets()
    resolved_db_url = db_url or os.getenv("DATABASE_URL")
    if not resolved_db_url:
        raise ValueError("DATABASE_URL is required for searching chunks.")

    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Please run `pip install psycopg2-binary`.")

    # Step 1: Embed question with Gemini Embedding API (gemini-embedding-001, RETRIEVAL_QUERY task_type)
    resolved_api_key = os.getenv("GEMINI_API_KEY")
    if not resolved_api_key or resolved_api_key.strip() in ("", "your_gemini_api_key_here"):
        raise ValueError("GEMINI_API_KEY is missing or invalid. Please set GEMINI_API_KEY in your .env file.")

    logger.info(f"Embedding search query with Gemini API: '{question}'")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=resolved_api_key)
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=384
        )
    )
    if not res or not res.embeddings:
        raise RuntimeError("Failed to generate embedding for query using Gemini API.")
    
    question_embedding = res.embeddings[0].values

    # Step 2: Query PostgreSQL with pgvector <=> cosine distance operator
    logger.info(f"Searching top {top_k} similar code chunks for repo: {repo_url}...")
    conn = psycopg2.connect(resolved_db_url)
    retrieved_chunks = []
    try:
        if register_vector is not None:
            register_vector(conn)

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            query = """
                SELECT 
                    id,
                    repo_url,
                    file_path,
                    start_line,
                    end_line,
                    chunk_text,
                    language,
                    (embedding <=> %s::vector) AS distance
                FROM code_chunks
                WHERE repo_url = %s
                ORDER BY distance ASC
                LIMIT %s;
            """
            cur.execute(query, (question_embedding, repo_url, top_k))
            rows = cur.fetchall()

            for row in rows:
                distance = float(row["distance"])
                similarity = max(0.0, round(1.0 - distance, 4))
                retrieved_chunks.append({
                    "id": row["id"],
                    "file_path": row["file_path"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "chunk_text": row["chunk_text"],
                    "language": row["language"],
                    "distance": round(distance, 4),
                    "similarity": similarity
                })
    finally:
        conn.close()

    logger.info(f"Retrieved {len(retrieved_chunks)} relevant code chunks.")
    return retrieved_chunks


def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> str:
    """
    1. Formats context prompt containing retrieved code chunks with file paths and line ranges.
    2. Instructs Gemini to answer strictly using only the provided code context.
    3. Calls Google Gemini API (gemini-2.5-flash) and returns generated response text.
    """
    sync_secrets()
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key or resolved_api_key.strip() in ("", "your_gemini_api_key_here"):
        raise ValueError(
            "GEMINI_API_KEY is missing or invalid. Please set your API key in your .env file "
            "or obtain one from https://aistudio.google.com/app/apikey"
        )

    # Format context blocks
    if not retrieved_chunks:
        return "I don't have enough context to answer that because no relevant code chunks were found for this repository."

    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        block = (
            f"--- Code Chunk #{idx} ---\n"
            f"File: {chunk['file_path']} (Lines {chunk['start_line']}-{chunk['end_line']})\n"
            f"Language: {chunk['language']}\n\n"
            f"```\n{chunk['chunk_text']}\n```"
        )
        context_blocks.append(block)

    formatted_context = "\n\n".join(context_blocks)

    prompt = f"""You are an expert AI code assistant. Answer the user's question about the codebase using ONLY the code context provided below.

CRITICAL INSTRUCTIONS:
1. Base your answer STRICTLY on the code chunks provided in the CONTEXT section.
2. If the context does not contain enough information to answer the question, state clearly: "I don't have enough context to answer that."
3. Always reference specific file paths and line numbers when explaining code logic.

CONTEXT:
{formatted_context}

USER QUESTION:
{question}

ANSWER:"""

    logger.info("Calling Google Gemini API (gemini-2.5-flash) for answer generation...")

    from google import genai
    client = genai.Client(api_key=resolved_api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    if response and response.text:
        return response.text
    else:
        raise RuntimeError("Google Gemini API returned an empty response.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python query_engine.py <repo_url> <question>")
        sys.exit(1)

    test_repo_url = sys.argv[1]
    test_question = sys.argv[2]

    try:
        results = search_chunks(test_question, test_repo_url, top_k=3)
        print(f"\nFound {len(results)} chunks:")
        for c in results:
            print(f"- {c['file_path']}:{c['start_line']}-{c['end_line']} (similarity: {c['similarity']})")

        print("\nGenerating answer with Gemini...")
        ans = generate_answer(test_question, results)
        print("\n--- Answer ---")
        print(ans)
    except Exception as err:
        logger.error(f"Query engine error: {err}")
        sys.exit(1)
