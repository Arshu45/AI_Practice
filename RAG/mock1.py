from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()
PROMPT = "What is the capital of India ?"
TEMPERATURE = 0.7


def build_model(provider: str):
    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not found in environment")

        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=gemini_model, temperature=TEMPERATURE)

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not found in environment")

        try:
            from langchain.llms import Groq
        except ModuleNotFoundError:
            try:
                from langchain_groq import Groq
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Groq support is not installed. Install the Groq LangChain integration "
                    "or use MODEL_PROVIDER=gemini."
                ) from exc

        groq_model = os.getenv("GROQ_MODEL", "groq-1.2-mini")
        return Groq(model=groq_model, temperature=TEMPERATURE)

    raise RuntimeError(
        "Unsupported MODEL_PROVIDER %r. Use 'gemini' or 'groq'." % provider
    )


model = build_model(MODEL_PROVIDER)

prompt = PROMPT
result = model.invoke(prompt) if hasattr(model, "invoke") else model(prompt)

if hasattr(result, "content"):
    print(result.content)
else:
    print(result)
