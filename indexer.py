import os
import sys
import tempfile
import shutil
import logging
import warnings
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Suppress harmless HuggingFace transformers warning logs
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Third-party imports
try:
    import git
except ImportError:
    git = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import psycopg2
    import psycopg2.extras
    from pgvector.psycopg2 import register_vector
except ImportError:
    psycopg2 = None
    register_vector = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("CodeIndexer")

# Allowed extensions and language mapping
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
}

# Folders to completely ignore during scanning
EXCLUDED_FOLDERS = {
    ".git",
    "node_modules",
    "dist",
    "build",
    "venv",
    ".venv",
    "__pycache__",
}


def clone_repository(repo_url: str, dest_dir: str) -> None:
    """
    Clones a remote git repository into the specified directory.
    Uses GitPython if available, otherwise falls back to subprocess git clone.
    """
    logger.info(f"Cloning repository: {repo_url} into temp folder...")
    if git is not None:
        git.Repo.clone_from(repo_url, dest_dir, depth=1)
    else:
        import subprocess
        result = subprocess.run(["git", "clone", "--depth", "1", repo_url, dest_dir], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr}")
    logger.info("Repository cloned successfully.")


def filter_and_collect_files(repo_path: str, max_files: int = 500) -> List[Path]:
    """
    Scans the repository directory, filtering by supported extensions and skipping excluded folders.
    If file count exceeds max_files (500), prints a warning and returns the smallest 500 files.
    """
    repo_dir = Path(repo_path)
    collected_files: List[Path] = []

    for root, dirs, files in os.walk(repo_dir):
        # Prune excluded directory names in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_FOLDERS and not d.startswith(".")]

        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                collected_files.append(file_path)

    total_found = len(collected_files)
    logger.info(f"Found {total_found} supported source files.")

    if total_found > max_files:
        logger.warning(
            f"Repository exceeds file limit ({total_found} > {max_files}). "
            f"Sorting files by size and selecting the smallest {max_files} files."
        )
        collected_files.sort(key=lambda p: p.stat().st_size if p.exists() else 0)
        collected_files = collected_files[:max_files]

    return collected_files


def chunk_file(
    file_path: str,
    content: str,
    extension: str,
    chunk_size: int = 50,
    overlap: int = 10
) -> List[Dict[str, Any]]:
    """
    Splits file content into fixed line chunks with overlap.
    
    Args:
        file_path: Relative or display path of the file
        content: String content of the source code file
        extension: File extension (e.g. '.py')
        chunk_size: Number of lines per chunk (default 50)
        overlap: Number of lines to overlap between chunks (default 10)
        
    Returns:
        List of chunk dicts containing metadata and text
    """
    language = SUPPORTED_EXTENSIONS.get(extension.lower(), "unknown")
    lines = content.splitlines()
    total_lines = len(lines)

    if total_lines == 0:
        return []

    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size  # Fallback to prevent infinite loop if overlap >= chunk_size

    for start_idx in range(0, total_lines, step):
        end_idx = min(start_idx + chunk_size, total_lines)
        chunk_lines = lines[start_idx:end_idx]
        chunk_text = "\n".join(chunk_lines)

        # 1-indexed line numbers
        start_line = start_idx + 1
        end_line = end_idx

        chunks.append({
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "chunk_text": chunk_text,
            "language": language
        })

        if end_idx >= total_lines:
            break

    return chunks


def process_repository_files(repo_path: str, files: List[Path]) -> List[Dict[str, Any]]:
    """
    Reads files from the cloned repository and runs chunk_file on each.
    """
    repo_dir = Path(repo_path)
    all_chunks: List[Dict[str, Any]] = []

    for file_path in files:
        try:
            relative_path = file_path.relative_to(repo_dir).as_posix()
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            
            file_chunks = chunk_file(
                file_path=relative_path,
                content=content,
                extension=file_path.suffix
            )
            all_chunks.extend(file_chunks)
        except Exception as e:
            logger.warning(f"Skipping file {file_path} due to read error: {e}")

    logger.info(f"Processed {len(files)} files into {len(all_chunks)} text chunks.")
    return all_chunks


def generate_embeddings(chunks: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Uses sentence-transformers to generate 384-dim vector embeddings for each chunk's text in batch.
    """
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is not installed. Please run `pip install sentence-transformers`.")

    if not chunks:
        return []

    logger.info(f"Loading embedding model '{model_name}'...")
    model = SentenceTransformer(model_name)

    texts = [chunk["chunk_text"] for chunk in chunks]
    logger.info(f"Generating embeddings for {len(texts)} chunks in batches...")
    
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def setup_database(conn) -> None:
    """
    Ensures pgvector extension exists and creates code_chunks table if not already present.
    """
    with conn.cursor() as cur:
        logger.info("Setting up database schema (enabling pgvector extension & code_chunks table)...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS code_chunks (
                id SERIAL PRIMARY KEY,
                repo_url TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INT NOT NULL,
                end_line INT NOT NULL,
                chunk_text TEXT NOT NULL,
                language TEXT NOT NULL,
                embedding VECTOR(384)
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_code_chunks_repo_url ON code_chunks(repo_url);")
    conn.commit()


def store_chunks(repo_url: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]], db_url: str) -> None:
    """
    Deletes existing chunks for the given repo_url and batch-inserts the new chunks + embeddings.
    """
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Please run `pip install psycopg2-binary`.")

    logger.info("Connecting to PostgreSQL database...")
    conn = psycopg2.connect(db_url)
    try:
        # First ensure table and pgvector extension are created
        setup_database(conn)

        # Register vector type adapter after extension is guaranteed to exist
        if register_vector is not None:
            register_vector(conn)

        with conn.cursor() as cur:
            # Delete old entries for idempotency
            logger.info(f"Removing existing records for repo_url: {repo_url}...")
            cur.execute("DELETE FROM code_chunks WHERE repo_url = %s;", (repo_url,))
            
            if chunks:
                logger.info(f"Batch-inserting {len(chunks)} chunks into code_chunks table...")
                insert_query = """
                    INSERT INTO code_chunks (repo_url, file_path, start_line, end_line, chunk_text, language, embedding)
                    VALUES %s
                """
                records = [
                    (
                        repo_url,
                        chunk["file_path"],
                        chunk["start_line"],
                        chunk["end_line"],
                        chunk["chunk_text"],
                        chunk["language"],
                        emb
                    )
                    for chunk, emb in zip(chunks, embeddings)
                ]
                psycopg2.extras.execute_values(cur, insert_query, records, page_size=100)

        conn.commit()
        logger.info("All chunks and embeddings stored successfully.")
    finally:
        conn.close()


def index_repository(repo_url: str, db_url: Optional[str] = None) -> Dict[str, Any]:
    """
    Full end-to-end indexing pipeline:
    1. Clones repository into temporary folder
    2. Filters & collects source files (cap 500)
    3. Chunks files with 50-line chunks + 10-line overlap
    4. Generates batch embeddings (all-MiniLM-L6-v2)
    5. Stores vector chunks into PostgreSQL + pgvector
    """
    load_dotenv()
    
    resolved_db_url = db_url or os.getenv("DATABASE_URL")
    if not resolved_db_url:
        raise ValueError(
            "DATABASE_URL not found. Please set DATABASE_URL in your .env file "
            "or pass it as an argument."
        )

    logger.info(f"Starting indexing pipeline for: {repo_url}")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Clone repo
        clone_repository(repo_url, temp_dir)

        # Step 2: Collect & filter files
        files_to_process = filter_and_collect_files(temp_dir, max_files=500)

        if not files_to_process:
            logger.warning("No matching source files found to index.")
            return {"repo_url": repo_url, "files_processed": 0, "chunks_created": 0}

        # Step 3: Chunk files
        chunks = process_repository_files(temp_dir, files_to_process)

        # Step 4: Generate Embeddings
        embeddings = generate_embeddings(chunks)

        # Step 5: Store in database
        store_chunks(repo_url, chunks, embeddings, resolved_db_url)

    logger.info("Indexing pipeline completed successfully!")
    return {
        "repo_url": repo_url,
        "files_processed": len(files_to_process),
        "chunks_created": len(chunks)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python indexer.py <repo_url>")
        sys.exit(1)
        
    target_repo_url = sys.argv[1]
    try:
        summary = index_repository(target_repo_url)
        print("\n--- Indexing Summary ---")
        print(f"Repository: {summary['repo_url']}")
        print(f"Files Processed: {summary['files_processed']}")
        print(f"Chunks Stored: {summary['chunks_created']}")
    except Exception as err:
        logger.error(f"Indexing failed: {err}")
        sys.exit(1)
