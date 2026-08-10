from dotenv import load_dotenv
import os

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").strip().lower()

def main():
    if MODEL_PROVIDER == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not found in environment")

        gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        model = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.7)

    elif MODEL_PROVIDER == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY not found in environment")

        groq_model = os.getenv("GROQ_MODEL", "groq-1.2-mini")
        model = ChatGroq(model=groq_model, temperature=0.7)

    else:
        raise RuntimeError(
            f"Unsupported MODEL_PROVIDER {MODEL_PROVIDER}. Use 'gemini' or 'groq'."
        )

    prompt = "What is the capital of India?"
    result = model.invoke(prompt) if hasattr(model, "invoke") else model(prompt)

    if hasattr(result, "content"):
        print(result.content)
    else:
        print(result)


if __name__ == "__main__":
    main()

