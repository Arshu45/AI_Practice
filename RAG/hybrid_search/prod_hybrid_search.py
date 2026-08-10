"""
Building RAG Pipelines
Complete retrieval-augmented generation implementation
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

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