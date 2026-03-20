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
class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int

    strike_rate: float
    balls_per_boundary: float
    boundary_percentage: float
    summary: str


# Define functions
def calculate_strike_rate(state: BatsmanState) -> BatsmanState:
    strike_rate = (state["runs"] / state["balls"]) * 100
    return {'strike_rate': strike_rate}

def calculate_balls_per_boundary(state: BatsmanState) -> BatsmanState:
    total_boundaries = state["fours"] + state["sixes"]
    balls_per_boundary = state["balls"] / total_boundaries if total_boundaries > 0 else float('inf')
    return {'balls_per_boundary': balls_per_boundary}

def calculate_boundary_percentage(state: BatsmanState) -> BatsmanState:
    total_boundaries = state["fours"]*4 + state["sixes"]*6
    boundary_percentage = (total_boundaries / state["runs"]) * 100 if state["runs"] > 0 else 0
    return {'boundary_percentage': boundary_percentage}

def summary(state: BatsmanState) -> BatsmanState:
    summary_text = (
        f"Strike Rate: {state['strike_rate']:.2f}\n"
        f"Balls per Boundary: {state['balls_per_boundary']:.2f}\n"
        f"Boundary Percentage: {state['boundary_percentage']:.2f}%\n"
    )
    summary = summary_text
    return {'summary': summary}

# Define Graph
graph = StateGraph(BatsmanState)

# Add nodes to the graph
graph.add_node('calculate_strike_rate', calculate_strike_rate)
graph.add_node('calculate_balls_per_boundary', calculate_balls_per_boundary)
graph.add_node('calculate_boundary_percentage', calculate_boundary_percentage)
graph.add_node('summary', summary)



# Define edges (parallel execution)
graph.add_edge(START, 'calculate_strike_rate')
graph.add_edge(START, 'calculate_balls_per_boundary')
graph.add_edge(START, 'calculate_boundary_percentage')
graph.add_edge('calculate_strike_rate', 'summary')
graph.add_edge('calculate_balls_per_boundary', 'summary')
graph.add_edge('calculate_boundary_percentage', 'summary')
graph.add_edge('summary', END)


# Compile the graph
workflow = graph.compile()

# Visualize the graph
# print(workflow.get_graph().draw_mermaid())

# Execute the workflow
initial_state = {
    "runs": 120,
    "balls": 80,
    "fours": 10,
    "sixes": 5
}
final_state = workflow.invoke(initial_state)
print("\n" + "="*50)
# print("FINAL SUMMARY:\n", final_state)
print("FINAL SUMMARY:\n", final_state['summary'])
print("="*50)