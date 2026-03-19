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
class BlogState(TypedDict):
    title: str
    outline: str
    content: str


# Define functions to call LLM

# Function to create outline
def create_outline(state: BlogState) -> BlogState:
    title = state["title"]

    # Formulate the prompt:
    prompt = f"Create a detailed outline for a blog post on the topic of: {title}"

    response = model.invoke(prompt)
    state["outline"] = response.content
    return state

# Function to write content based on title and outline
def write_content(state: BlogState) -> BlogState:
    title = state["title"]
    outline = state["outline"]

    # Formulate the prompt:
    prompt = f"Write a detailed blog post based on the following title and outline:\n\nTitle: {title}\n\nOutline: {outline}"

    response = model.invoke(prompt)
    state["content"] = response.content
    return state

# Function to display the final result
def display_result(state):
    print("\n" + "="*50)
    print("TITLE:\n", state["title"])
    print("\n" + "="*50)
    print("OUTLINE:\n", state["outline"])
    print("\n" + "="*50)
    print("CONTENT:\n", state["content"])
    print("="*50)


# Create Graph
graph = StateGraph(BlogState)

# Add nodes to the graph
graph.add_node("create_outline", create_outline)
graph.add_node("write_content", write_content)

# Add Edges to the graph
graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "write_content")
graph.add_edge("write_content", END)

# Compile the graph
workflow = graph.compile()


# Execute the workflow  
initial_state = {"title": "The Benefits of Exercise"} # Initial state with title
final_state = workflow.invoke(initial_state)
display_result(final_state)

# print(f"Final State: {final_state}")

# Visualize the graph
# print(workflow.get_graph().draw_mermaid())
