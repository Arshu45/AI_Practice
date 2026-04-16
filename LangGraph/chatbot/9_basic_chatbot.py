# Chatbot 

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Safety check (important)
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY not found in environment")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)


# Define State
class ChatState(TypedDict):

    # We need to use a Reducer function to accumulate the conversation history in the state, so we will store the history of messages in the state itself. This way we can provide the entire conversation history as context to the model for generating more coherent and contextually relevant responses.
    messages: Annotated[list[BaseMessage], add_messages]


# Define function to call LLM
def chat(state: ChatState) -> ChatState:
    # prompt
    messages = state["messages"]
    
    # invoke model
    response = model.invoke(messages)

    # update state with new message
    return {"messages": [response]}

# Stores the conversation history in memory, allowing the chatbot to maintain context across interactions. This is crucial for generating coherent and contextually relevant responses, as the model can refer back to previous messages in the conversation.
checkpoint = MemorySaver()

# Define Graph
graph = StateGraph(ChatState)

# Add Nodes
graph.add_node('chat', chat)

# Define edges
graph.add_edge(START, 'chat')
graph.add_edge('chat', END)

# Compile Graph , passing the checkpoint to enable memory saving of the conversation history. This allows the chatbot to maintain context across interactions, making the conversation more coherent and relevant.
chatbot = graph.compile(checkpointer=checkpoint)   

# Visualize the graph
# print(chatbot.get_graph().draw_mermaid())


# Example Usage₹
# initial_state = {
#     "messages": [
#         SystemMessage(content="You are a helpful and friendly assistant."),
#         HumanMessage(content="Hello! Can you tell me a joke?")
#     ]
# }

# final_state = chatbot.invoke(initial_state)
# # Print the conversation history
# # print(final_state["messages"])
# for message in final_state["messages"]:
#     print(f"{message.__class__.__name__}: {message.content}")   

thread_id = 1

while True:
    user_input = input("Type your message (or 'exit' to quit): ")

    if user_input.strip().lower() in ['exit', 'quit', 'bye']:
        print("Goodbye!")
        break

    config = {'configurable': {'thread_id': thread_id}}

    # Add user message to the conversation history
    response = chatbot.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    # Print the assistant's response
    print(f"AI: {response['messages'][-1].content}")

    
print(chatbot.get_state(config=config))