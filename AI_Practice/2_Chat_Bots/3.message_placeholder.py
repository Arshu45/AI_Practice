# What is a Message Placeholder?

# A Message Placeholder is a slot inside a chat prompt where a list of messages will be injected later. Instead of inserting one string, you insert many messages (conversation history, tool messages, scratchpad, etc.).

# Why do we need it?

# LLMs (chat models) think in messages, not just text:

# system
# user
# assistant
# tool

# When building agents, memory, or multi-turn chat, you don’t know in advance: how many messages there will be, what roles they’ll have, So LangChain gives you MessagePlaceholder as a dynamic container.


from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()


from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Chat template
chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are helpful dietition who recommends when and what to eat."),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# print(chat_template)

chat_history = [
    HumanMessage(content="I want to gain weight"),
    AIMessage(content="You should eat more calories than you burn."),
    HumanMessage(content="What should I eat?"),
    AIMessage(content="You should eat a mix of protein, carbs, and fats."),
    HumanMessage(content="I ate eggs yesterday!!, is that good ?")
]

# print(chat_template.format_messages(chat_history=chat_history, input="What should I eat today?"))


# # Load Chat History
# chat_history = []
# with open(r"C:\Users\arsha\Desktop\Langchain_tutorials\chatbot_history.txt", "r") as file:
#     chat_history.extend(file.readlines())

prompt = chat_template.invoke({
    "chat_history": chat_history,
    'input': "What should I eat today?"
})

result = model.invoke(prompt)
print(parser.invoke(result.content))