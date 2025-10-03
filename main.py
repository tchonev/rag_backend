import os
import time
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple

import torch
import psycopg2
import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
embedding_model: Optional[SentenceTransformer] = None
llm_model = None
llm_tokenizer = None

# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------
class ContextRequest(BaseModel):
    query: str = Field(..., description="User search query")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")


class ContextResponse(BaseModel):
    strategy: str = Field(..., description="Retrieval strategy used")
    document_count: int = Field(..., description="Number of documents in context")
    chunk_count: int = Field(..., description="Number of chunks in context")
    selected_document: Optional[str] = Field(None, description="Document title if full document strategy")
    context: str = Field(..., description="Formatted context for language model")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata")


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    context_override: Optional[ContextResponse] = None


class QueryResponse(BaseModel):
    query: str
    context: str
    response: str
    metadata: Dict[str, Any]


# -----------------------------------------------------------------------------
# DB config
# -----------------------------------------------------------------------------
DATABASE_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "user": os.getenv("DB_USER", "ragtask"),
    "password": os.getenv("DB_PASSWORD", "mypassword"),
    "database": os.getenv("DB_NAME", "vectordb"),
}


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(**DATABASE_CONFIG, cursor_factory=psycopg2.extras.RealDictCursor)


# -----------------------------------------------------------------------------
# App lifecycle
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager"""
    global embedding_model, llm_model, llm_tokenizer

    logger.info("Loading models...")
    try:
        # Embedding model
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Embedding model loaded")

        # TinyLlama
        logger.info("Loading TinyLlama...")
        llm_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        llm_model = AutoModelForCausalLM.from_pretrained(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Ensure pad token
        if llm_tokenizer.pad_token_id is None:
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        logger.info(f"TinyLlama loaded on: {next(llm_model.parameters()).device}")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield
    logger.info("Application shutting down...")


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Document Context Retrieval API",
    description="Semantic search with intelligent document retrieval for RAG applications",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Domain objects
# -----------------------------------------------------------------------------
class DocumentChunk:
    """Internal representation of a document chunk"""

    def __init__(
        self,
        title: str,
        uuid: str,
        content: str,
        chunk_index: int,
        page_numbers: str,
        industries: List[str],
        country_codes: List[str],
        chunk_length: int,
        similarity_score: float,
    ):
        self.title = title
        self.uuid = uuid
        self.content = content
        self.chunk_index = chunk_index
        self.page_numbers = page_numbers
        self.industries = industries or []
        self.country_codes = country_codes or []
        self.chunk_length = chunk_length
        self.similarity_score = similarity_score


# -----------------------------------------------------------------------------
# Retrieval helpers
# -----------------------------------------------------------------------------
def search_similar_chunks(query: str, similarity_threshold: float) -> List[DocumentChunk]:
    """Step 1: Find chunks above similarity threshold"""
    if not embedding_model:
        raise HTTPException(status_code=500, detail="Embedding model not loaded")

    query_embedding = embedding_model.encode([query])[0].tolist()

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 
                title,
                uuid,
                content,
                chunk_index,
                page_numbers,
                industries,
                country_codes,
                chunk_length,
                1 - (embedding <=> %s::vector) as similarity_score
            FROM document_chunks 
            WHERE embedding IS NOT NULL 
              AND 1 - (embedding <=> %s::vector) > %s
            ORDER BY embedding <=> %s::vector
        """,
            (query_embedding, query_embedding, similarity_threshold, query_embedding),
        )

        rows = cur.fetchall()
        chunks: List[DocumentChunk] = []
        for row in rows:
            chunks.append(
                DocumentChunk(
                    title=row["title"],
                    uuid=row["uuid"],
                    content=row["content"],
                    chunk_index=row["chunk_index"],
                    page_numbers=row.get("page_numbers") or "",
                    industries=row.get("industries") or [],
                    country_codes=row.get("country_codes") or [],
                    chunk_length=row["chunk_length"],
                    similarity_score=row["similarity_score"],
                )
            )
        return chunks
    finally:
        conn.close()


def get_full_document(title: str) -> List[DocumentChunk]:
    """Fetch all chunks for a complete document"""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 
                title,
                uuid,
                content,
                chunk_index,
                page_numbers,
                industries,
                country_codes,
                chunk_length
            FROM document_chunks 
            WHERE title = %s
            ORDER BY chunk_index
        """,
            (title,),
        )

        rows = cur.fetchall()
        chunks: List[DocumentChunk] = []
        for row in rows:
            chunks.append(
                DocumentChunk(
                    title=row["title"],
                    uuid=row["uuid"],
                    content=row["content"],
                    chunk_index=row["chunk_index"],
                    page_numbers=row.get("page_numbers") or "",
                    industries=row.get("industries") or [],
                    country_codes=row.get("country_codes") or [],
                    chunk_length=row["chunk_length"],
                    similarity_score=0.0,
                )
            )
        return chunks
    finally:
        conn.close()


def determine_strategy(chunks: List[DocumentChunk]) -> Tuple[str, Optional[str], List[DocumentChunk]]:
    """Step 2: Determine retrieval strategy"""
    if not chunks:
        return "no_matches", None, []

    # Group chunks by title
    doc_groups: Dict[str, List[DocumentChunk]] = defaultdict(list)
    for c in chunks:
        doc_groups[c.title].append(c)

    # Full document candidates: >2 chunks; pick highest average similarity
    full_doc_candidates: Dict[str, float] = {}
    for title, doc_chunks in doc_groups.items():
        if len(doc_chunks) > 2:
            avg_sim = sum(x.similarity_score for x in doc_chunks) / len(doc_chunks)
            full_doc_candidates[title] = avg_sim

    if full_doc_candidates:
        selected_title = max(full_doc_candidates, key=full_doc_candidates.get)
        return "full_document", selected_title, []

    return "individual_chunks", None, chunks


def format_context(chunks: List[DocumentChunk]) -> str:
    """Step 3: Format context for language model"""
    if not chunks:
        return ""

    doc_groups: Dict[str, List[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        doc_groups[chunk.title].append(chunk)

    parts: List[str] = []

    for title, doc_chunks in doc_groups.items():
        doc_chunks.sort(key=lambda x: x.chunk_index)
        industries_str = ", ".join(doc_chunks[0].industries) if doc_chunks[0].industries else "None"
        countries_str = ", ".join(doc_chunks[0].country_codes) if doc_chunks[0].country_codes else "None"

        header = f"Document: {title} (Industries: {industries_str} | Countries: {countries_str})"
        parts.append(header)
        parts.append("=" * len(header))

        for chunk in doc_chunks:
            if chunk.page_numbers:
                parts.append(f"[Pages: {chunk.page_numbers}]")
            parts.append(chunk.content)
            parts.append("")

        parts.append("---")
        parts.append("")

    return "\n".join(parts).strip()


def truncate_chunks_by_tokens(chunks: List[DocumentChunk], max_tokens: int, strategy: str) -> List[DocumentChunk]:
    """
    Truncate chunks to fit within token budget.

    Strategy-aware:
    - full_document: Keep chunks in order, truncate from END (preserve document flow)
    - individual_chunks: Keep highest similarity chunks (already sorted by similarity)
    """
    if not chunks or not llm_tokenizer:
        return chunks

    # Reserve tokens for document headers and formatting
    OVERHEAD_PER_DOC = 50  # "Document: X (Industries: ...) \n===\n"
    OVERHEAD_PER_CHUNK = 20  # "[Pages: X]\n\n---\n"

    # For full_document: chunks are in sequential order, keep from start
    # For individual_chunks: chunks are already sorted by similarity, iterate as-is

    selected_chunks: List[DocumentChunk] = []
    total_tokens = 0
    doc_titles_seen = set()

    for chunk in chunks:
        # Estimate overhead for new document header
        overhead = 0
        if chunk.title not in doc_titles_seen:
            overhead += OVERHEAD_PER_DOC
            doc_titles_seen.add(chunk.title)
        overhead += OVERHEAD_PER_CHUNK

        # Count tokens in chunk content
        chunk_tokens = len(llm_tokenizer.encode(chunk.content, add_special_tokens=False))

        if total_tokens + chunk_tokens + overhead <= max_tokens:
            selected_chunks.append(chunk)
            total_tokens += chunk_tokens + overhead
        else:
            # Budget exhausted
            if strategy == "full_document" and not selected_chunks:
                # Special case: if even first chunk doesn't fit, truncate it
                available_tokens = max_tokens - overhead
                if available_tokens > 100:  # Minimum useful size
                    tokens = llm_tokenizer.encode(chunk.content, add_special_tokens=False)[:available_tokens]
                    truncated_content = llm_tokenizer.decode(tokens, skip_special_tokens=True)
                    chunk.content = truncated_content + "\n[...truncated]"
                    selected_chunks.append(chunk)
            break

    if strategy == "full_document":
        logger.info(
            f"Token truncation (full_document): kept first {len(selected_chunks)}/{len(chunks)} chunks (~{total_tokens} tokens)")
    else:
        logger.info(
            f"Token truncation (individual_chunks): kept top {len(selected_chunks)}/{len(chunks)} chunks by similarity (~{total_tokens} tokens)")

    return selected_chunks


def retrieve_context_logic(query: str, similarity_threshold: float, max_context_tokens: int) -> ContextResponse:
    """Pure function used by both endpoints (and by tests)"""
    start_time = time.time()

    similar_chunks = search_similar_chunks(query, similarity_threshold)
    strategy, selected_document, final_chunks = determine_strategy(similar_chunks)

    if strategy == "full_document" and selected_document:
        final_chunks = get_full_document(selected_document)

    # Token-aware truncation (respects strategy)
    if llm_tokenizer is not None:
        final_chunks = truncate_chunks_by_tokens(final_chunks, max_context_tokens, strategy)

    context = format_context(final_chunks)
    processing_time = (time.time() - start_time) * 1000.0
    doc_count = len({c.title for c in final_chunks}) if final_chunks else 0
    avg_similarity = (
        sum(c.similarity_score for c in similar_chunks) / len(similar_chunks)
    ) if similar_chunks else 0.0

    #This is a quick and dirty fix to avoid overloading the model with too big a context and hanging
    # if len(context) > 6000:  # Roughly 2000 tokens
    #     context = context[:6000] + "\n[Context truncated due to length]"


    return ContextResponse(
        strategy=strategy,
        document_count=doc_count,
        chunk_count=len(final_chunks),
        selected_document=selected_document,
        context=context,
        metadata={
            "similarity_threshold": similarity_threshold,
            "initial_matches": len(similar_chunks),
            "avg_similarity": round(avg_similarity, 3),
            "processing_time_ms": round(processing_time, 1),
            "context_length": len(context),
        },
    )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        conn.close()
        db_connected = True
    except Exception:
        db_connected = False

    return {
        "status": "healthy",
        "embedding_model_loaded": embedding_model is not None,
        "llm_loaded": llm_model is not None and llm_tokenizer is not None,
        "db_connected": db_connected,
    }


@app.post("/retrieve-context", response_model=ContextResponse, tags=["Context Retrieval"])
def retrieve_context(request: ContextRequest):
    """Retrieve context for a query using intelligent document retrieval strategy."""
    try:
        logger.info(f"Processing context retrieval for query: '{request.query}'")
        return retrieve_context_logic(request.query, request.similarity_threshold)
    except Exception as e:
        logger.error(f"Error in context retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Query Processing"])
async def query_endpoint(request: QueryRequest):
    """
    Answer a query using context from retrieval.
    Context is pre-truncated by token count to fit model capacity.
    """
    global llm_model, llm_tokenizer

    if llm_model is None or llm_tokenizer is None:
        raise HTTPException(status_code=500, detail="LLM not loaded")

    logger.info(f"=== Starting query: {request.query[:50]}...")

    MODEL_MAX_TOKENS = 2048
    RESERVE_FOR_SYSTEM_PROMPT = 180  # System instructions + formatting
    RESERVE_FOR_GENERATION = 400

    # 1) Measure actual query length
    query_tokens = len(llm_tokenizer.encode(request.query, add_special_tokens=False))
    logger.info(f"Query tokens: {query_tokens}")

    # 2) Calculate dynamic context budget
    MAX_CONTEXT_TOKENS = MODEL_MAX_TOKENS - RESERVE_FOR_SYSTEM_PROMPT - query_tokens - RESERVE_FOR_GENERATION

    # Safety check - ensure we have reasonable room for context
    if MAX_CONTEXT_TOKENS < 200:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long ({query_tokens} tokens). Maximum query length is ~{MODEL_MAX_TOKENS - RESERVE_FOR_SYSTEM_PROMPT - RESERVE_FOR_GENERATION - 200} tokens."
        )

    logger.info(f"Token budget - System: {RESERVE_FOR_SYSTEM_PROMPT}, Query: {query_tokens}, Context: {MAX_CONTEXT_TOKENS}, Generation: {RESERVE_FOR_GENERATION}")


    # 1) Context: use override if provided, else compute with token limit
    if request.context_override is not None:
        context_resp = request.context_override
    else:
        logger.info("Retrieving context...")
        context_resp = retrieve_context_logic(
            request.query,
            request.similarity_threshold,
            max_context_tokens=MAX_CONTEXT_TOKENS
        )
        logger.info(f"Context retrieved: {len(context_resp.context)} chars, {context_resp.chunk_count} chunks")

    context = context_resp.context
    metadata = context_resp.metadata
    chunk_count = context_resp.chunk_count
    avg_similarity = float(metadata.get("avg_similarity", 0.0))

    # 2) Build prompt
    system_prompt = """You are a helpful AI assistant. You will be provided with context from knowledge base and a user question.

INSTRUCTIONS:
- Use the provided context to answer the question when it contains relevant information
- If the context doesn't contain enough information to fully answer the question, say so clearly
- Don't make up information that isn't in the context
- If the context is only partially relevant, use what's helpful and indicate what you cannot answer
- Be concise but thorough in your response
- If the context seems unrelated to the question, politely indicate this
"""

    context_quality_note = ""
    if avg_similarity < 0.7:
        context_quality_note = f"\nNOTE: The retrieved context has moderate relevance (similarity: {avg_similarity:.2f}). Use it carefully."
    elif chunk_count < 2:
        context_quality_note = f"\nNOTE: Only {chunk_count} relevant document chunk(s) were found. The answer may be incomplete."

    prompt = f"""{system_prompt}

CONTEXT:
{context}
{context_quality_note}

QUESTION: {request.query}

ANSWER:"""

    logger.info(f"Prompt built: {len(prompt)} chars")

    # 3) Tokenize prompt
    logger.info("Tokenizing prompt...")
    device = next(llm_model.parameters()).device
    tok_out = llm_tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=False
    )
    input_ids = tok_out["input_ids"].to(device)
    attention_mask = tok_out["attention_mask"].to(device)

    prompt_len = int(input_ids.shape[-1])
    logger.info(f"Tokenization complete: {prompt_len} tokens")

    # 4) Calculate available tokens for generation
    tokens_available_for_generation = MODEL_MAX_TOKENS - prompt_len
    max_new_tokens = min(RESERVE_FOR_GENERATION, tokens_available_for_generation)

    logger.info(
        f"Prompt: {prompt_len} tokens, Generation budget: {max_new_tokens} tokens, Available: {tokens_available_for_generation}")

    # Safety check
    if max_new_tokens < 50:
        raise HTTPException(
            status_code=500,
            detail=f"Prompt too long ({prompt_len} tokens). Only {max_new_tokens} tokens left for generation."
        )

    # 5) Generate answer
    logger.info("Starting generation...")
    with torch.no_grad():
        outputs = llm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=llm_tokenizer.pad_token_id,
            eos_token_id=llm_tokenizer.eos_token_id,
        )
    logger.info("Generation complete!")

    # 6) Decode
    generated = outputs[0][prompt_len:]
    answer = llm_tokenizer.decode(generated, skip_special_tokens=True).strip()

    if answer.startswith("ANSWER:"):
        answer = answer[7:].strip()

    response_tokens = int(generated.shape[-1])
    logger.info(f"Response: {response_tokens} tokens")

    return QueryResponse(
        query=request.query,
        context=context,
        response=answer,
        metadata={
            "chunk_count": chunk_count,
            "avg_similarity": avg_similarity,
            "similarity_threshold": request.similarity_threshold,
            "context_length": len(context),
            "prompt_tokens": prompt_len,
            "response_tokens": response_tokens,
            "total_tokens": prompt_len + response_tokens,
            **metadata,
        },
    )


@app.get("/stats", tags=["Statistics"])
def get_stats():
    """Get database statistics"""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT title) as total_documents,
                AVG(chunk_length) as avg_chunk_length
            FROM document_chunks
        """
        )
        stats = cur.fetchone()
        return {
            "total_chunks": stats["total_chunks"],
            "total_documents": stats["total_documents"],
            "avg_chunk_length": round(float(stats["avg_chunk_length"]), 1) if stats["avg_chunk_length"] is not None else 0.0,
            "embedding_model": "all-MiniLM-L6-v2",
        }
    finally:
        conn.close()
