# langchain_core.prompts is used to define reusable, structured prompt templates that dynamically inject user input before sending requests to an LLM.

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

chat_template = ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert!"),
    ('human', "Explain in simple terms the concept of {topic}")
])

prompt = chat_template.invoke({
    'domain': 'Geography',
    'topic': 'Where does tropic of cancer lies ?'
})


print(prompt)
result = model.invoke(prompt)
print(result.content)