from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
import os

load_dotenv()

# Safety check (important)
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY not found in environment")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

# Define State
class LLMState(TypedDict):
    question: str
    answer: str


# Define function to call LLM
def llm_qa(state: LLMState) -> LLMState:
    question = state["question"]

    # Formulate the prompt:
    prompt = f"Answer the following question: {question}"

    response = model.invoke(prompt)
    state["answer"] = response.content
    return state

# Create Graph
graph = StateGraph(LLMState)

# Add nodes to the graph
graph.add_node("llm_qa", llm_qa)

# Add Edges to the graph
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)

# Compile the graph
workflow = graph.compile()


# Execute the workflow  
initial_state = {"question": "What is the capital of New Zealand?"} # Initial state with question
final_state = workflow.invoke(initial_state)
print(f"Final State: {final_state}")

# Visualize the graph
# print(workflow.get_graph().draw_mermaid())
