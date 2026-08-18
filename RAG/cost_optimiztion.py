"""
Cost Optimization Patterns
Reducing LLM costs in production using free/open-tier models (Gemini / Groq)
"""

import os
import hashlib
import json
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

load_dotenv()

# Determine active model provider from environment (default: gemini)
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()


def get_llm(model: Optional[str] = None, temperature: float = 0):
    """
    Initialize and return a chat model dynamically based on MODEL_PROVIDER (.env).
    Follows the pattern in rag_pipeline.py.
    """
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
        # Fallback to Gemini
        return init_chat_model(
            model=model or "gemini-2.5-flash",
            model_provider="google_genai",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
        )


# === 1. Model Routing ===


class ModelRouter:
    """Route queries to appropriate model based on complexity."""

    def __init__(self):
        if MODEL_PROVIDER == "groq":
            self.cheap_model_name = "llama-3.1-8b-instant"
            self.expensive_model_name = "llama-3.3-70b-versatile"
            self.cheap_cost_per_1k = 0.00005  # ~$0.05 / 1M tokens
            self.expensive_cost_per_1k = 0.00059  # ~$0.59 / 1M tokens
        else:
            # Gemini provider
            self.cheap_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.expensive_model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            self.cheap_cost_per_1k = 0.000075  # ~$0.075 / 1M tokens
            self.expensive_cost_per_1k = 0.00030

        self.cheap_model = get_llm(model=self.cheap_model_name, temperature=0)
        self.expensive_model = get_llm(model=self.expensive_model_name, temperature=0)
        self.classifier = get_llm(model=self.cheap_model_name, temperature=0)

    def classify_complexity(self, query: str) -> str:
        """Classify query complexity into 'simple' or 'complex'."""
        prompt = ChatPromptTemplate.from_template(
            """
Classify this query's complexity as 'simple' or 'complex'.

Simple: Basic facts, short answers, simple calculations
Complex: In-depth analysis, extensive reasoning, multi-step problem solving

Query: {query}

Respond with only: simple or complex
"""
        )

        response = self.classifier.invoke(prompt.format(query=query))
        classification = response.content.strip().lower()
        return "complex" if "complex" in classification else "simple"

    @traceable(name="routed_query")
    def invoke(self, query: str) -> Tuple[str, str, float]:
        """
        Route and invoke query based on complexity classification.
        Returns: (response_text, model_name, estimated_cost)
        """
        complexity = self.classify_complexity(query)

        if complexity == "simple":
            model = self.cheap_model
            model_name = self.cheap_model_name
            cost_per_1k = self.cheap_cost_per_1k
        else:
            model = self.expensive_model
            model_name = self.expensive_model_name
            cost_per_1k = self.expensive_cost_per_1k

        response = model.invoke(query)

        # Estimate cost based on rough token count
        tokens = len(query.split()) * 1.3
        estimated_cost = (tokens / 1000) * cost_per_1k

        return response.content, model_name, estimated_cost


def demo_model_routing():
    """Demonstrate model routing based on query complexity."""
    print("=" * 60)
    print(f"1. Model Routing Demo (Provider: {MODEL_PROVIDER.upper()}):")
    print("=" * 60)

    router = ModelRouter()

    queries = [
        "What is 2 + 2?",  # Simple
        "Analyze the economic and workforce implications of generative AI over the next decade.",  # Complex
        "What color is the sky?",  # Simple
    ]

    total_cost = 0.0
    for query in queries:
        result, model, cost = router.invoke(query)
        total_cost += cost
        print(f"\nQuery: {query}")
        print(f"  Routed Model: {model}")
        print(f"  Est. Cost:    ${cost:.6f}")
        preview = result.strip().replace("\n", " ")[:80] + "..."
        print(f"  Response:     {preview}")

    print(f"\nTotal Estimated Cost: ${total_cost:.6f}\n")


# === 2. Exact & Semantic Caching ===


class SemanticCache:
    """Cache responses with normalized hashing & exact matching."""

    def __init__(self):
        self.cache: Dict[str, Dict[str, str]] = {}

    def _hash_query(self, query: str) -> str:
        """Create MD5 hash of normalized query."""
        normalized = " ".join(query.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        """Get cached response if query exists."""
        query_hash = self._hash_query(query)
        if query_hash in self.cache:
            return self.cache[query_hash]["response"]
        return None

    def set(self, query: str, response: str):
        """Cache a response."""
        query_hash = self._hash_query(query)
        self.cache[query_hash] = {"query": query, "response": response}

    def stats(self) -> dict:
        return {"cached_queries": len(self.cache)}


class CachedLLM:
    """LLM wrapper with response caching."""

    def __init__(self):
        self.llm = get_llm(temperature=0)
        self.cache = SemanticCache()
        self.cache_hits = 0
        self.cache_misses = 0

    @traceable(name="cached_invoke")
    def invoke(self, query: str) -> Tuple[str, bool]:
        """
        Invoke LLM with caching.
        Returns: (response, from_cache)
        """
        cached = self.cache.get(query)
        if cached:
            self.cache_hits += 1
            return cached, True

        # Cache miss - call model
        self.cache_misses += 1
        response = self.llm.invoke(query)
        result = response.content

        # Save to cache
        self.cache.set(query, result)
        return result, False

    def get_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
        }


def demo_caching():
    """Demonstrate query caching."""
    print("=" * 60)
    print("2. LLM Caching Demo:")
    print("=" * 60)

    llm = CachedLLM()

    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",  # Cache hit (exact)
        "what is python?  ",  # Cache hit (normalized)
        "What is Rust?",
    ]

    for query in queries:
        result, from_cache = llm.invoke(query)
        source = "CACHE HIT " if from_cache else "LLM CALL  "
        preview = result.strip().replace("\n", " ")[:60] + "..."
        print(f"[{source}] '{query}' -> {preview}")

    print(f"\nCache Stats: {llm.get_stats()}\n")


# === 3. Token Budgeting ===


class TokenBudget:
    """Track and enforce token limits across requests."""

    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_per_request = max_tokens_per_request
        self.usage = {"total_input": 0, "total_output": 0, "requests": 0}

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (~1.3 tokens per word)."""
        return int(len(text.split()) * 1.3)

    def check_budget(self, text: str) -> Tuple[bool, int]:
        """Check if request is within budget."""
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage."""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:
        total = self.usage["total_input"] + self.usage["total_output"]
        return {
            **self.usage,
            "total_tokens": total,
            "avg_per_request": total / max(self.usage["requests"], 1),
        }


class BudgetedLLM:
    """LLM wrapper with proactive token budgeting guardrails."""

    def __init__(self, max_tokens: int = 4000):
        self.llm = get_llm(temperature=0)
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    @traceable(name="budgeted_invoke")
    def invoke(self, query: str) -> str:
        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {tokens} tokens > limit of {self.budget.max_per_request}"
            )

        response = self.llm.invoke(query)
        result = response.content

        output_tokens = self.budget.estimate_tokens(result)
        self.budget.record_usage(tokens, output_tokens)

        return result

    def get_stats(self) -> dict:
        return self.budget.get_stats()


def demo_token_budgeting():
    """Demonstrate token budget enforcement."""
    print("=" * 60)
    print("3. Token Budgeting Demo:")
    print("=" * 60)

    llm = BudgetedLLM(max_tokens=60)

    queries = [
        "What is AI in one sentence?",  # Within budget
        "Explain " + "very " * 80 + "complex quantum mechanics in full detail",  # Over budget
        "Give me three programming languages.",  # Within budget
    ]

    for query in queries:
        try:
            result = llm.invoke(query)
            preview = result.strip().replace("\n", " ")[:60] + "..."
            print(f"✅ ACCEPTED: '{query[:35]}...' -> {preview}")
        except ValueError as e:
            print(f"❌ REJECTED: '{query[:35]}...' -> {e}")

    print(f"\nUsage Stats: {llm.get_stats()}\n")


if __name__ == "__main__":
    demo_model_routing()
    demo_caching()
    demo_token_budgeting()