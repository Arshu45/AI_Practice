"""
Production-Ready Chunking Pipeline
Hybrid & Fallback-Resilient Chunking for Enterprise RAG
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Determine active embedding provider (matches rag_pipeline.py and embeddings.py)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()


def get_embedding_model():
    """Get the active embedding model based on MODEL_PROVIDER (.env)."""
    if MODEL_PROVIDER == "gemini":
        class GeminiEmbedding2Wrapper(GoogleGenerativeAIEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self.embed_query(text) for text in texts]
        return GeminiEmbedding2Wrapper(model="gemini-embedding-2")
    elif MODEL_PROVIDER == "groq":
        return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    else:
        class GeminiEmbedding2Wrapper(GoogleGenerativeAIEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self.embed_query(text) for text in texts]
        return GeminiEmbedding2Wrapper(model="gemini-embedding-2")


embeddings = get_embedding_model()


# Sample document with 3 distinct topics (Machine Learning -> Cooking -> Astronomy)
MULTI_TOPIC_TEXT = """
Machine learning models learn patterns from historical data to make accurate predictions on unseen inputs.
Supervised learning requires large labeled datasets where each sample has an associated ground truth target.
Neural networks pass input signals through interconnected layers of artificial neurons to capture complex non-linear relationships.
Gradient descent iteratively updates model weights to minimize the loss function and improve accuracy.

Making authentic Neapolitan pizza requires finely ground double zero flour, filtered water, fresh yeast, and sea salt.
The dough must undergo a slow fermentation process of twenty-four to forty-eight hours to develop complex flavor and airy crust structure.
San Marzano tomatoes grown in the volcanic soil near Mount Vesuvius provide the ideal sweet and acidic sauce base.
Baking in a wood-fired oven at nine hundred degrees Fahrenheit creates signature leopard-spotting on the crust within ninety seconds.

Mars is the fourth planet from the Sun and the second-smallest planet in the Solar System.
Its distinctive reddish appearance is caused by iron oxide rust pervasive across its dusty surface.
NASA's Perseverance rover is currently exploring Jezero Crater searching for biosignatures of ancient microbial life.
Liquid water cannot exist permanently on the Martian surface due to low atmospheric pressure less than one percent of Earth's.
""".strip()


class ProductionChunker:
    """
    Production-grade chunker with:
    1. Semantic chunking as primary strategy for topic preservation.
    2. Automatic fallback to RecursiveCharacterTextSplitter on failure or short texts.
    3. Secondary splitting for any oversized semantic chunks that exceed max token/char limits.
    4. Rich metadata enrichment (chunk index, character count, method used).
    """

    def __init__(
        self,
        embedding_model=None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 85.0,
        target_chunk_size: int = 500,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 50,
        min_text_length: int = 100,
    ):
        self.embeddings = embedding_model or embeddings
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_text_length = min_text_length

        # Initialize fallback recursive splitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.target_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Initialize semantic splitter
        try:
            self.semantic_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            )
        except Exception as e:
            logger.warning(f"Failed to initialize SemanticChunker: {e}. Falling back to RecursiveSplitter.")
            self.semantic_splitter = None

    def chunk_text(
        self,
        text: str,
        use_semantic: bool = True,
    ) -> List[str]:
        """
        Split a raw text string into chunks with fallback resilience.
        """
        if not text or not text.strip():
            return []

        cleaned_text = text.strip()

        # If text is too short, return as a single chunk
        if len(cleaned_text) <= self.min_text_length:
            return [cleaned_text]

        raw_chunks: List[str] = []
        strategy_used = "recursive"

        # Attempt Semantic Chunking if requested
        if use_semantic and self.semantic_splitter is not None:
            try:
                raw_chunks = self.semantic_splitter.split_text(cleaned_text)
                strategy_used = "semantic"
            except Exception as e:
                logger.warning(f"Semantic chunking failed ({e}). Falling back to RecursiveCharacterTextSplitter.")
                raw_chunks = self.recursive_splitter.split_text(cleaned_text)
                strategy_used = "recursive_fallback"
        else:
            raw_chunks = self.recursive_splitter.split_text(cleaned_text)
            strategy_used = "recursive_direct"

        # Guardrail: Check for any oversized chunks and perform secondary splitting
        final_chunks: List[str] = []
        for chunk in raw_chunks:
            if len(chunk) > self.max_chunk_size:
                sub_chunks = self.recursive_splitter.split_text(chunk)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)

        return final_chunks

    def chunk_documents(
        self,
        documents: List[Document],
        use_semantic: bool = True,
    ) -> List[Document]:
        """
        Chunk a list of LangChain Document objects, preserving and enriching their metadata.
        """
        chunked_docs: List[Document] = []

        for doc_idx, doc in enumerate(documents):
            text_chunks = self.chunk_text(doc.page_content, use_semantic=use_semantic)
            total_chunks = len(text_chunks)

            for chunk_idx, chunk_text in enumerate(text_chunks):
                # Clone existing metadata and append chunking lineage
                metadata = dict(doc.metadata) if doc.metadata else {}
                metadata.update(
                    {
                        "doc_index": doc_idx,
                        "chunk_index": chunk_idx,
                        "total_chunks": total_chunks,
                        "char_count": len(chunk_text),
                        "estimated_tokens": int(len(chunk_text.split()) * 1.3),
                        "strategy": "semantic" if use_semantic else "recursive",
                    }
                )

                chunked_docs.append(
                    Document(page_content=chunk_text, metadata=metadata)
                )

        return chunked_docs


def smart_chunker(
    text: str,
    use_semantic: bool = True,
    fallback_chunk_size: int = 500,
    max_chunk_size: int = 1000,
) -> List[str]:
    """
    Production chunking function with semantic as primary and recursive as fallback.

    Args:
        text (str): Input document text.
        use_semantic (bool): If True, uses embedding-based semantic boundaries.
        fallback_chunk_size (int): Chunk size for recursive fallback.
        max_chunk_size (int): Hard maximum character limit per chunk.

    Returns:
        List[str]: Cleaned, bounded text chunks.
    """
    chunker = ProductionChunker(
        target_chunk_size=fallback_chunk_size,
        max_chunk_size=max_chunk_size,
    )
    return chunker.chunk_text(text, use_semantic=use_semantic)


# =====================================================================
# Demonstrations & Validation
# =====================================================================

def demo_smart_chunker():
    """Demonstrate smart_chunker on multi-topic text."""
    print("=" * 75)
    print("1. SMART CHUNKER (Semantic Primary)")
    print("=" * 75)

    start_time = time.time()
    chunks = smart_chunker(MULTI_TOPIC_TEXT, use_semantic=True)
    elapsed = time.time() - start_time

    print(f"Generated {len(chunks)} chunks in {elapsed:.2f}s:\n")
    for i, c in enumerate(chunks):
        preview = c.strip().replace("\n", " ")
        if len(preview) > 110:
            preview = preview[:110] + "..."
        print(f"[Chunk {i + 1}] ({len(c)} chars, ~{len(c.split())} words):")
        print(f"  {preview}\n")


def demo_production_document_chunking():
    """Demonstrate ProductionChunker with metadata tracking."""
    print("=" * 75)
    print("2. PRODUCTION DOCUMENT CHUNKER (With Metadata Enrichment)")
    print("=" * 75)

    sample_docs = [
        Document(
            page_content=MULTI_TOPIC_TEXT,
            metadata={"source": "tech_culinary_space_guide.md", "author": "Arsh"},
        ),
        Document(
            page_content="LangChain and LangGraph provide modular abstractions for building LLM agents.",
            metadata={"source": "short_note.txt", "author": "Arsh"},
        ),
    ]

    chunker = ProductionChunker()
    processed_chunks = chunker.chunk_documents(sample_docs, use_semantic=True)

    print(f"Processed {len(sample_docs)} document(s) into {len(processed_chunks)} chunk(s):\n")
    for i, doc in enumerate(processed_chunks):
        print(f"--- Chunk {i + 1} ---")
        print(f"Metadata: {doc.metadata}")
        snippet = doc.page_content.strip().replace("\n", " ")[:100] + "..."
        print(f"Content:  {snippet}\n")


def demo_fallback_resilience():
    """Demonstrate fast recursive fallback mode."""
    print("=" * 75)
    print("3. FALLBACK / FAST MODE (Recursive Character Splitting)")
    print("=" * 75)

    chunks = smart_chunker(MULTI_TOPIC_TEXT, use_semantic=False, fallback_chunk_size=400)

    print(f"Generated {len(chunks)} chunks using recursive fallback:\n")
    for i, c in enumerate(chunks):
        snippet = c.strip().replace("\n", " ")[:90] + "..."
        print(f"[Chunk {i + 1}] ({len(c)} chars): {snippet}")


if __name__ == "__main__":
    demo_smart_chunker()
    demo_production_document_chunking()
    demo_fallback_resilience()