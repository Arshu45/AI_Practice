"""
Building RAG Pipelines
Complete retrieval-augmented generation implementation
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chat_models import init_chat_model

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import tempfile


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

embeddings_model = get_embedding_model()

# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""

def get_llm():
    if MODEL_PROVIDER == "gemini":
        return init_chat_model(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            model_provider="google_genai",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )
    elif MODEL_PROVIDER == "groq":
        return init_chat_model(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            model_provider="groq",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    else:
        return init_chat_model(
            model="gemini-2.5-flash",
            model_provider="google_genai",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0,
        )

llm = get_llm()


def create_kb():
    """Create a vector store from knowledge base."""

    # Step 1: Split the knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    doc = Document(
        page_content=KNOWLEDGE_BASE, metadata={"source": "langchain_knowledge_base.md"}
    )
    chunks = splitter.split_documents([doc])

    # Step 2: Creating vectors and loading them into Chroma DB
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )
    return vector_store

def demo_basic_rag():

    vector_store = create_kb()
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 2}
    )

    # RAG Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the following context:

{context}

Question: {question}

Answer:


Make sure to answer in a concise manner, 
and if you don't know the answer, just say "I don't know."""
    )

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    # Rag chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Test the RAG chain
    # Test
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?",
    ]

    print("Basic RAG Demo:\n")
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


if __name__ == "__main__":
    demo_basic_rag()
