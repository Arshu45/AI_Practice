from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Schema
class Review(BaseModel):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

prompt = "Provide a summary and sentiment of the following review: 'The product was excellent and met all my expectations.'"

response = structured_model.invoke(prompt)
print(response)  # {'summary': 'The product was excellent and met all my expectations.', 'sentiment': 'positive'}
