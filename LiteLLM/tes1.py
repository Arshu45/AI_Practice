import litellm
import os
from dotenv import load_dotenv

load_dotenv()

# For LiteLLM proxy
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")  # Set this in your .env file
LITELLM_BASE_URL = "https://smartpal.cybage.com/litellm/v1"

# Set your API key (replace with your actual key)
# For OpenAI: os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
# For other providers, check the docs for required environment variables

# os.environ["OPENAI_API_KEY"] = os.getenv("LITE_LLM_API_KEY")


# Test with LiteLLM proxy
def test_litellm_proxy():
    if not LITELLM_API_KEY:
        print("Please set LITELLM_API_KEY in your .env file")
        return
    
    try:
        response = litellm.completion(
            model="openai/claude-haiku-4.5",  # Prefix with openai/ for proxy compatibility
            api_base=LITELLM_BASE_URL,
            api_key=LITELLM_API_KEY,
            messages=[{"role": "user", "content": "What is the capital of France?"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "capital_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "country": {"type": "string"},
                            "capital": {"type": "string"}
                        },
                        "required": ["country", "capital"],
                        "additionalProperties": False
                    }
                }
            }
        )
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Uncomment the function you want to run
    test_litellm_proxy()
    pass
