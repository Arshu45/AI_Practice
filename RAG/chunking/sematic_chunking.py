"""
Semantic Chunking in LangChain
Understanding Context-Aware Document Chunking for RAG
"""

import os
import re
import numpy as np
from dotenv import load_dotenv
from typing import List, Tuple

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

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


# =====================================================================
# 1. UNDER THE HOOD: How Semantic Chunking Works Step-by-Step
# =====================================================================

def cosine_distance(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine distance between two vectors: (1 - cosine_similarity)."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1.0 - float(similarity)


def demo_semantic_chunking_from_scratch():
    """
    Step-by-step demonstration of the mathematical mechanics:
    1. Split text into individual sentences.
    2. Embed each sentence into high-dimensional vector space.
    3. Calculate cosine distance between adjacent sentence pairs.
    4. Detect threshold spikes (breakpoints) where topic shifts occur.
    5. Group sentences between breakpoints into semantic chunks.
    """
    print("=" * 75)
    print("1. UNDER THE HOOD: Semantic Chunking Mechanics (Step-by-Step)")
    print("=" * 75)

    # Step 1: Split into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', MULTI_TOPIC_TEXT) if s.strip()]
    print(f"\n[Step 1] Extracted {len(sentences)} sentences from text.")

    # Step 2: Generate embeddings for each sentence
    print("[Step 2] Generating vector embeddings for all sentences...")
    sentence_vectors = [embeddings.embed_query(s) for s in sentences]

    # Step 3: Compute distances between adjacent sentences (Sentence i vs Sentence i+1)
    print("[Step 3] Computing Cosine Distance between consecutive sentences:\n")
    distances = []
    for i in range(len(sentence_vectors) - 1):
        dist = cosine_distance(sentence_vectors[i], sentence_vectors[i + 1])
        distances.append(dist)

    # Display distance chart
    print(f"{'Pair':<8} {'Distance':<10} {'Transition / Topic Shift Preview'}")
    print("-" * 75)
    for i, dist in enumerate(distances):
        preview_a = sentences[i][:28] + "..."
        preview_b = sentences[i + 1][:28] + "..."
        # ASCII bar graph representation
        bar = "█" * int(dist * 30)
        print(f"S{i+1}->S{i+2:<4} {dist:.4f}  {bar:<12} '{preview_a}' -> '{preview_b}'")

    # Step 4: Calculate Breakpoint Threshold (e.g., 80th percentile)
    threshold = float(np.percentile(distances, 80))
    print(f"\n[Step 4] Breakpoint Threshold (80th Percentile): {threshold:.4f}")
    print("  * Distances ABOVE this threshold represent major topic shifts (Cut Points).")
    print("  * Distances BELOW this threshold represent continuous context (Group Together).")

    # Step 5: Split at breakpoints
    chunks = []
    current_chunk = [sentences[0]]

    for i, dist in enumerate(distances):
        if dist > threshold:
            # Major topic shift detected -> finalize current chunk and start new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            # Same topic -> append to current chunk
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"\n[Step 5] Final Semantic Chunks Formed ({len(chunks)} chunks):")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ({len(chunk.split())} words) ---")
        print(chunk)


# =====================================================================
# 2. LANGCHAIN INTEGRATION: Using SemanticChunker with Threshold Modes
# =====================================================================

def demo_langchain_semantic_chunker():
    """
    Demonstrate LangChain's built-in SemanticChunker with different breakpoint types:
    - percentile: Splits at top X% distance spikes (default: 95)
    - standard_deviation: Splits when distance > mean + (X * std_dev)
    - interquartile: Splits using IQR statistical outliers
    - gradient: Splits based on raw distance gradients
    """
    print("\n" + "=" * 75)
    print("2. LANGCHAIN BUILT-IN: SemanticChunker Threshold Strategies")
    print("=" * 75)

    threshold_strategies = [
        ("percentile", {"breakpoint_threshold_amount": 80.0}),
        ("standard_deviation", {"breakpoint_threshold_amount": 1.2}),
        ("interquartile", {"breakpoint_threshold_amount": 1.5}),
    ]

    for strategy_name, kwargs in threshold_strategies:
        print(f"\n>>> Strategy: '{strategy_name}' (params: {kwargs})")
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=strategy_name,
            **kwargs
        )

        docs = chunker.create_documents([MULTI_TOPIC_TEXT])
        print(f"    Result: Split into {len(docs)} chunk(s)")
        for idx, doc in enumerate(docs):
            snippet = doc.page_content.strip().replace("\n", " ")[:90] + "..."
            print(f"      [Chunk {idx + 1}] ({len(doc.page_content)} chars): {snippet}")


# =====================================================================
# 3. COMPARISON: Traditional vs Semantic Chunking
# =====================================================================

def demo_comparison():
    """
    Side-by-side comparison:
    Fixed-size RecursiveCharacterTextSplitter vs SemanticChunker
    """
    print("\n" + "=" * 75)
    print("3. COMPARISON: Fixed-Size vs Semantic Chunking")
    print("=" * 75)

    # 1. Traditional Character Chunking
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    char_chunks = char_splitter.split_text(MULTI_TOPIC_TEXT)

    print(f"\n[A] Traditional Fixed-Size (chunk_size=400, overlap=50) -> {len(char_chunks)} Chunks:")
    for i, c in enumerate(char_chunks):
        preview = c.strip().replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:100] + "..."
        print(f"  Chunk {i+1} ({len(c)} chars): {preview}")

    # 2. Semantic Chunking
    semantic_chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80.0
    )
    semantic_chunks = semantic_chunker.split_text(MULTI_TOPIC_TEXT)

    print(f"\n[B] Semantic Chunking (Percentile=80) -> {len(semantic_chunks)} Topical Chunks:")
    for i, c in enumerate(semantic_chunks):
        preview = c.strip().replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:100] + "..."
        print(f"  Chunk {i+1} ({len(c)} chars): {preview}")


if __name__ == "__main__":
    demo_semantic_chunking_from_scratch()
    demo_langchain_semantic_chunker()
    demo_comparison()
