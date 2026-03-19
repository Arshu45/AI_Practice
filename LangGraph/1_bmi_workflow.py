from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from IPython.display import Image


# Define State
class BMIState(TypedDict):
    weight: float  # in kilograms
    height: float  # in meters
    bmi: float  # Body Mass Index
    category: str  # BMI Category


def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight"]
    height = state["height"]
    bmi = weight / (height ** 2)
    state["bmi"] = round(bmi, 2)
    return state

def categorize_bmi(state: BMIState) -> BMIState:
    bmi = state["bmi"]
    if bmi < 18.5:
        state["category"] = "Underweight"
    elif 18.5 <= bmi < 24.9:
        state["category"] = "Normal weight"
    elif 25 <= bmi < 29.9:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obesity"
    return state


# Define Graph
graph = StateGraph(BMIState) 


# Add nodes to the graph
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("categorize_bmi", categorize_bmi)

# Add Edges to the graph
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "categorize_bmi")
graph.add_edge("categorize_bmi", END)

# Compile the graph
workflow = graph.compile()

# Execute the workflow
initial_state = {"weight": 59, "height": 1.67} # Initial state with weight and height
final_state = workflow.invoke(initial_state)
print(f"Final State: {final_state}")



# Visualize the graph
print(workflow.get_graph().draw_mermaid())
