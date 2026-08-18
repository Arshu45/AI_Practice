"""
Building RAG Pipelines
Complete retrieval-augmented generation implementation
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:
    from langchain_community.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma

from dotenv import load_dotenv
import numpy as np


load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()

def get_embedding_model():
    if MODEL_PROVIDER == "gemini":
        class GeminiEmbedding2Wrapper(GoogleGenerativeAIEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self.embed_query(text) for text in texts]
        return GeminiEmbedding2Wrapper(model="gemini-embedding-2")
    elif MODEL_PROVIDER == "groq":
        return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    else:
        # Fallback to gemini as default
        class GeminiEmbedding2Wrapper(GoogleGenerativeAIEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [self.embed_query(text) for text in texts]
        return GeminiEmbedding2Wrapper(model="gemini-embedding-2")

embeddings = get_embedding_model()


documents = [
    Document(
        page_content="Product SKU-7742X is a high-quality widget designed for efficiency and durability.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="Product SKU-8821Y is an innovative gadget that enhances user experience with advanced features.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Product SKU-9933Z is a versatile tool that combines functionality with sleek design, making it a must-have for professionals.",
        metadata={"source": "vector_guide", "topic": "database"},
    )
]

vector_store = Chroma.from_documents(documents, embeddings, collection_name="hybrid_test")

vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

bm25_retriever = BM25Retriever.from_documents(documents, k=2)

# Combine the two retrievers into an ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])  # Adjust weights as needed


def test_query(query, name, retriever):
    print(f"--- {name} ---")
    results = retriever.invoke(query)
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        print(f"Result {i + 1}: {preview} (Source: {doc.metadata.get('source', 'N/A')})")
    return results

test_queries = [
    "Tell me about Product SKU-7742X.",
    "What are the features of Product SKU-8821Y?",
    "Can you provide details on Product SKU-9933Z?"
]

for query in test_queries:
    print(f"\nQuery: {query}")
    test_query(query, "Vector Store Retriever", vector_retriever)
    test_query(query, "BM25 Retriever", bm25_retriever)
    test_query(query, "Ensemble Retriever", ensemble_retriever)
    

        
