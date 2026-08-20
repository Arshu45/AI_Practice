"""
Advanced RAG Patterns
Multi-Query, Contextual Compression, Hybrid Search (Ensemble), and Parent Document Retrieval
Compatible with multi-provider free models (Gemini / Groq / OpenAI)
"""

import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

# LangChain core & splitters
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector store & document store
from langchain_chroma import Chroma

# Model providers
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Advanced retrievers (with fallback support)
try:
    from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    from langchain_classic.retrievers import (
        ContextualCompressionRetriever,
        EnsembleRetriever,
        ParentDocumentRetriever,
    )
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
    from langchain_classic.storage import InMemoryStore
except ImportError:
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.retrievers import (
        ContextualCompressionRetriever,
        EnsembleRetriever,
        ParentDocumentRetriever,
    )
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.storage import InMemoryStore

from langchain_community.retrievers import BM25Retriever

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Determine active model and embedding provider from .env
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


def get_llm(model: Optional[str] = None, temperature: float = 0):
    """Get the active LLM based on MODEL_PROVIDER (.env)."""
    if MODEL_PROVIDER == "gemini":
        default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return init_chat_model(
            model=model or default_model,
            model_provider="google_genai",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )
    elif MODEL_PROVIDER == "groq":
        default_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return init_chat_model(
            model=model or default_model,
            model_provider="groq",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )
    else:
        return init_chat_model(
            model=model or "gemini-2.5-flash",
            model_provider="google_genai",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )


embeddings = get_embedding_model()


# =====================================================================
# Sample Knowledge Bases for Demos
# =====================================================================

INFO_BURIED = [
    Document(
        page_content="""ACME AI SOLUTIONS - COMPANY HISTORY AND TECHNOLOGY STACK

Founded in 2018 by three Stanford graduates, ACME AI Solutions began as a
small consulting firm helping enterprises adopt machine learning. Our first
office was a converted garage in Palo Alto, and we had just two laptops and
a dream. The early days were challenging - we survived on instant ramen and
the occasional pizza from the client meetings.

In 2019, we secured our first major contract with a Fortune 500 retailer,
helping them build a recommendation engine. This led to rapid growth and we
moved to a proper office space in San Francisco. By 2020, we had grown to
50 employees and opened offices in Austin and Seattle.

Our current technology stack has evolved significantly over the years. For
backend services, we use Python and FastAPI. Our data pipeline runs on
Apache Spark and Airflow. For frontend, we've standardized on React and
TypeScript.

LangChain is a framework for building LLM applications. It provides tools
for prompts, chains, agents, and memory. LangChain supports multiple LLM
providers including OpenAI, Anthropic, and local models like Llama.

The company culture at ACME emphasizes work-life balance. We offer unlimited
PTO, which most employees use for an average of 25 days per year. Our
engineering teams follow agile methodology with two-week sprints.

Our revenue has grown consistently, from $2M in 2019 to $45M in 2023. We
project $70M for 2024, driven by our new enterprise AI platform. The company
went through Series B funding in 2022, raising $80M at a $500M valuation.

Employee benefits include comprehensive health insurance through Aetna, a
401(k) with 4% matching, and a generous equity package.""",
        metadata={"source": "acme_company_overview.pdf"},
    ),
    Document(
        page_content="""ACME AI PLATFORM - TECHNICAL DOCUMENTATION v2.4

Chapter 1: System Architecture Overview

The ACME AI Platform is built on a microservices architecture deployed on
AWS EKS (Elastic Kubernetes Service). Each microservice is containerized
using Docker and orchestrated by Kubernetes. We use Istio as our service
mesh for traffic management and observability.

Our database layer consists of PostgreSQL for transactional data, Redis
for caching, and Pinecone for vector storage. All databases are deployed
in high-availability configurations with automatic failover.

Chapter 2: Authentication and Authorization

User authentication is handled through Auth0, supporting both SSO via SAML
2.0 and OAuth 2.0 flows. We implement role-based access control (RBAC) with
four default roles: Admin, Developer, Analyst, and Viewer.

Chapter 3: AI Framework Integration

LangGraph is a library for building stateful, multi-actor applications with
LLMs. Key features include state management, cycles and loops, human-in-the-
loop workflows, and persistence. LangGraph extends LangChain for complex
agent architectures.

Chapter 4: Monitoring and Logging

We use DataDog for application performance monitoring (APM) and log
aggregation. All services emit structured JSON logs that are collected and
indexed for searching. Alert thresholds are configured for latency (p99 >
500ms), error rates (> 1%), and resource utilization (CPU > 80%).

Chapter 5: Disaster Recovery

Our disaster recovery plan includes daily database backups stored in S3
with cross-region replication. RTO is 4 hours, and RPO is 1 hour.""",
        metadata={"source": "technical_docs_v2.4.pdf"},
    ),
]

TECH_DOCS = [
    Document(
        page_content="Python is a high-level programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, artificial intelligence, and automation.",
        metadata={
            "topic": "programming",
            "language": "python",
            "difficulty": "beginner",
        },
    ),
    Document(
        page_content="JavaScript is the language of the web. It runs in browsers and on servers with Node.js. Modern frameworks like React, Vue, and Angular make building interactive web applications efficient. JavaScript supports asynchronous programming with Promises and async/await.",
        metadata={
            "topic": "programming",
            "language": "javascript",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Machine learning is a subset of AI that enables systems to learn from data. Supervised learning uses labeled data, while unsupervised learning finds patterns in unlabeled data. Popular ML frameworks include TensorFlow, PyTorch, and scikit-learn.",
        metadata={
            "topic": "ai",
            "subtopic": "machine_learning",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="LangChain is a framework for building LLM applications. It provides tools for prompts, chains, agents, and memory. LangChain supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features include state management, cycles and loops, human-in-the-loop workflows, and persistence. LangGraph extends LangChain for complex agent architectures.",
        metadata={
            "topic": "ai",
            "subtopic": "llm_frameworks",
            "difficulty": "advanced",
        },
    ),
    Document(
        page_content="Docker is a platform for containerizing applications. Containers package code and dependencies together for consistent deployment. Docker Compose orchestrates multi-container applications. Kubernetes scales Docker containers in production.",
        metadata={
            "topic": "devops",
            "subtopic": "containers",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="PostgreSQL is an advanced open-source relational database. It supports JSON data types, full-text search, and extensions like pgvector for vector similarity search. PostgreSQL is ACID compliant and highly extensible.",
        metadata={
            "topic": "database",
            "type": "relational",
            "difficulty": "intermediate",
        },
    ),
    Document(
        page_content="Vector databases like Pinecone, Chroma, and Qdrant are optimized for storing and searching embeddings. They enable semantic similarity search for RAG applications. Most support metadata filtering and hybrid search combining keywords with vectors.",
        metadata={"topic": "database", "type": "vector", "difficulty": "intermediate"},
    ),
]


def create_base_vectorstore(collection_name: str = "tech_docs_base"):
    """Create an isolated in-memory vector store for demos."""
    return Chroma.from_documents(
        documents=TECH_DOCS,
        embedding=embeddings,
        collection_name=collection_name,
    )


# =====================================================================
# Pattern 1: Multi-Query Retriever
# =====================================================================

def demo_multi_query_retriever():
    """
    Multi-Query Retriever uses an LLM to generate multiple perspectives/paraphrases
    of the user's question, overcoming vocabulary mismatches between user queries and docs.
    """
    print("=" * 65)
    print("1. MULTI-QUERY RETRIEVER")
    print("Generates multiple query variations for higher recall")
    print("=" * 65)

    vectorstore = create_base_vectorstore("multi_query_demo")
    llm = get_llm(temperature=0.3)

    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        llm=llm,
    )

    query = "What tools can I use to build AI applications?"

    print(f"\nOriginal Query: '{query}'")
    print("\nGenerating variations and retrieving documents...")

    docs = retriever.invoke(query)

    print(f"\nRetrieved {len(docs)} unique document(s):")
    for i, doc in enumerate(docs):
        topic = doc.metadata.get("topic", "N/A")
        subtopic = doc.metadata.get("subtopic", "")
        tag = f"{topic}/{subtopic}" if subtopic else topic
        print(f"  {i+1}. [{tag}] {doc.page_content[:110]}...")


# =====================================================================
# Pattern 2: Contextual Compression Retriever
# =====================================================================

def demo_contextual_compression():
    """
    Contextual Compression uses an LLM to extract only the sentences
    relevant to the query, stripping out irrelevant noise from retrieved chunks.
    """
    print("\n" + "=" * 65)
    print("2. CONTEXTUAL COMPRESSION RETRIEVER")
    print("Extracts only query-relevant sentences from documents")
    print("=" * 65)

    # Use long documents where relevant info is buried in noise
    vectorstore = Chroma.from_documents(
        documents=INFO_BURIED,
        embedding=embeddings,
        collection_name="compression_demo",
    )
    llm = get_llm(temperature=0)

    # Create LLM-powered compressor
    compressor = LLMChainExtractor.from_llm(llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    )

    query = "What are the company's annual revenue numbers and projections?"

    print(f"\nQuery: '{query}'")

    # 1. Without Compression
    base_docs = vectorstore.as_retriever(search_kwargs={"k": 1}).invoke(query)
    print(f"\n--- WITHOUT Compression (Full Raw Chunk) ---")
    for doc in base_docs:
        print(f"Raw Length: {len(doc.page_content)} characters")
        print(f"Snippet:    {doc.page_content[:200]}...\n")

    # 2. With Compression
    compressed_docs = compression_retriever.invoke(query)
    print(f"--- WITH Compression (Cleaned Relevant Extract) ---")
    for doc in compressed_docs:
        print(f"Extracted Length: {len(doc.page_content)} characters")
        print(f"Content:          {doc.page_content.strip()}\n")


# =====================================================================
# Pattern 3: Ensemble / Hybrid Search (BM25 + Dense Vectors)
# =====================================================================

def demo_ensemble_hybrid_search():
    """
    Hybrid search combines keyword matching (BM25 for exact terms/acronyms)
    with dense vector search (semantic similarity for conceptual queries).
    """
    print("\n" + "=" * 65)
    print("3. ENSEMBLE / HYBRID RETRIEVER")
    print("Combines Keyword Search (BM25) + Semantic Vector Search (Chroma)")
    print("=" * 65)

    vectorstore = create_base_vectorstore("hybrid_demo")

    # 1. BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(TECH_DOCS)
    bm25_retriever.k = 3

    # 2. Dense semantic retriever
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. Ensemble combining both with weighted scoring
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6],  # 40% BM25, 60% Dense Vector
    )

    queries = [
        "ACID transactions",  # Keyword-specific (BM25 excels)
        "How do I store AI model outputs for later retrieval?",  # Conceptual (Dense Vector excels)
        "fast similarity lookup for embeddings",  # Mixed
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)

        bm25_results = bm25_retriever.invoke(query)
        semantic_results = semantic_retriever.invoke(query)
        ensemble_results = ensemble_retriever.invoke(query)

        bm25_top = bm25_results[0].page_content[:65] if bm25_results else "None"
        semantic_top = semantic_results[0].page_content[:65] if semantic_results else "None"
        ensemble_top = ensemble_results[0].page_content[:65] if ensemble_results else "None"

        print(f"  BM25 Top Result:     {bm25_top}...")
        print(f"  Semantic Top Result: {semantic_top}...")
        print(f"  Ensemble Top Result: {ensemble_top}...")


# =====================================================================
# Pattern 4: Parent Document Retriever (Small-to-Big)
# =====================================================================

def demo_parent_document_retriever():
    """
    Parent Document Retriever:
    Indexes small child chunks for pinpoint vector search,
    but returns the entire parent document/section for full LLM reasoning context.
    """
    print("\n" + "=" * 65)
    print("4. PARENT DOCUMENT RETRIEVER")
    print("Small child chunks for search -> Full parent documents for context")
    print("=" * 65)

    long_doc = Document(
        page_content="""# Complete Guide to Building AI Agents

## Chapter 1: Introduction to AI Agents
AI agents are autonomous systems that can perceive their environment, make decisions, and take actions to achieve goals. Unlike simple chatbots, agents can use tools, maintain state, and execute multi-step plans.
The key components of an AI agent include:
- A language model for reasoning
- Tools for interacting with external systems
- Memory for maintaining context
- A planning mechanism for complex tasks

## Chapter 2: Agent Frameworks
Several frameworks exist for building AI agents:
LangChain provides the foundational abstractions for chains and simple agents. It excels at straightforward tool-calling patterns and integrates with many LLM providers.
LangGraph extends LangChain for complex, stateful agents. It introduces graph-based state management, enabling cycles, human-in-the-loop workflows, and persistent execution.
CrewAI focuses on multi-agent collaboration, allowing teams of specialized agents to work together on complex tasks.

## Chapter 3: Production Considerations
Deploying agents to production requires careful attention to:
- Error handling and fallbacks
- Token usage optimization
- Observability and tracing
- Security and access control
- State persistence and recovery

LangSmith provides observability for LangChain/LangGraph applications, offering tracing, evaluation, and monitoring capabilities.""",
        metadata={"source": "ai_agents_guide.md"},
    )

    # Splitters: Parent (Big) and Child (Small)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

    # Storage
    vectorstore = Chroma(
        collection_name="parent_child_demo",
        embedding_function=embeddings,
    )
    store = InMemoryStore()

    # Create ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents([long_doc])

    query = "What is LangGraph used for?"
    print(f"\nQuery: '{query}'")

    # What the vector search matched (Child Chunk)
    child_docs = vectorstore.similarity_search(query, k=1)
    if child_docs:
        print(f"\n--- [1] Child Chunk Matched in Vector DB ({len(child_docs[0].page_content)} chars) ---")
        print(f"Content: {child_docs[0].page_content.strip()}")

    # What the retriever returns to the application (Parent Document)
    parent_docs = retriever.invoke(query)
    if parent_docs:
        print(f"\n--- [2] Parent Document Returned to LLM ({len(parent_docs[0].page_content)} chars) ---")
        print(f"Content:\n{parent_docs[0].page_content.strip()[:350]}...\n")


# =====================================================================
# Pattern 5: Complete Advanced RAG Chain (Multi-Query + Compression + QA)
# =====================================================================

def demo_advanced_rag_chain():
    """
    End-to-end production RAG chain:
    MultiQueryRetriever -> ContextualCompression -> Prompt -> LLM Generation
    """
    print("\n" + "=" * 65)
    print("5. COMPLETE ADVANCED RAG CHAIN")
    print("Multi-Query + Contextual Compression + Generation")
    print("=" * 65)

    vectorstore = create_base_vectorstore("rag_chain_demo")
    llm = get_llm(temperature=0)

    # Step 1: Multi-query for high recall
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        llm=llm,
    )

    # Step 2: Contextual compression to strip noise
    compressor = LLMChainExtractor.from_llm(llm)
    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_retriever,
    )

    # Step 3: Prompt template
    prompt = ChatPromptTemplate.from_template(
        """Answer the question based ONLY on the following context. Be specific and concise.

Context:
{context}

Question: {question}

Answer:"""
    )

    def format_docs(docs):
        return "\n\n".join(f"- {doc.page_content.strip()}" for doc in docs)

    # Step 4: Assemble RAG Chain (LCEL)
    rag_chain = (
        {"context": advanced_retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "What options do I have for building AI agents?",
        "How can I store and search embeddings?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = rag_chain.invoke(q)
        print(f"A: {answer.strip()}")


# =====================================================================
# Execution Entry Point
# =====================================================================

if __name__ == "__main__":
    # Choose which demo to run (uncomment to test):
    
    # 1. Multi-Query Retriever (Paraphrased query variations)
    # demo_multi_query_retriever()

    # 2. Contextual Compression (Extracts needle facts from noise)
    # demo_contextual_compression()

    # 3. Ensemble / Hybrid Search (BM25 Keyword + Dense Vector)
    # demo_ensemble_hybrid_search()

    # 4. Parent Document Retriever (Small-to-Big Retrieval)
    # demo_parent_document_retriever()

    # 5. Complete Advanced RAG Chain (Multi-Query + Compression + Generation)
    demo_advanced_rag_chain()