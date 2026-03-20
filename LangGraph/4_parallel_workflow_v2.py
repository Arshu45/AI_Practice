# UPSC Essay Workflow. That evaluates an essay on three parameters - language, analysis and clarity. Each parameter is evaluated in parallel and then the final evaluation is done based on the individual feedback and scores.

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import operator

load_dotenv()

# Safety check (important)
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY not found in environment")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

class EvaluationSchema(BaseModel):
    feedback: str = Field(..., description="Detailed feedback on the essay, highlighting strengths and areas for improvement.")
    score: int = Field(..., description="A score out of 10 evaluating the overall quality of the essay, considering factors such as content, structure, coherence, and language use.", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

essay = """
Rohit Sharma, widely known as "The Hitman," is celebrated as one of the most naturally gifted cricketers in the history of the sport. Born in Nagpur, he evolved from a talented off-spinner into a world-class opening batter, eventually becoming the only player to smash three double-centuries in ODI cricket. Beyond his effortless pull shots and record-breaking sixes, his legacy is defined by his calm leadership, having guided the Mumbai Indians to five IPL titles and captaining India to a historic T20 World Cup victory in 2024. As he continues his journey in 2026, Rohit remains an icon of elegance and resilience, leaving an indelible mark on the global game.
"""

prompt = f"Evaluate the language quality of the following essay and provide a feedback along with a score out of 10:\n\n{essay}"

# response = structured_model.invoke(prompt)
# print("\nDetailed Feedback:\n", response.feedback)
# print("\nOverall Score:", response.score)


# Define State
class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    average_score: float


# Define functions to call LLM
def evaluate_language(state: UPSCState) -> UPSCState:
    essay = state["essay"]

    prompt = f"Evaluate the language quality of the following essay and provide a feedback along with a score out of 10:\n\n{essay}"

    response = structured_model.invoke(prompt)
    return {
        'language_feedback': response.feedback,
        'individual_scores': [response.score]
    }

def evaluate_analysis(state: UPSCState) -> UPSCState:
    essay = state["essay"]

    prompt = f"Evaluate the depth of analysis in the following essay and provide a feedback along with a score out of 10:\n\n{essay}"

    response = structured_model.invoke(prompt)
    return {
        'analysis_feedback': response.feedback,
        'individual_scores': [response.score]
    }

def evaluate_clarity(state: UPSCState) -> UPSCState:
    essay = state["essay"]

    prompt = f"Evaluate the clarity of thought of the following essay and provide a feedback along with a score out of 10:\n\n{essay}"

    response = structured_model.invoke(prompt)
    return {
        'clarity_feedback': response.feedback,
        'individual_scores': [response.score]
    }

def final_evaluation(state: UPSCState) -> UPSCState:
    # Prompt
    prompt = f"Based on the following feedback create a summerized feedback \n language feedback: {state['language_feedback']} \n analysis feedback: {state['analysis_feedback']} \n clarity feedback: {state['clarity_feedback']} \n Also calculate the average score based on the individual scores: {state['individual_scores']}"
    response = model.invoke(prompt).content

    # Calculate average score
    average_score = sum(state['individual_scores']) / len(state['individual_scores']) if state['individual_scores'] else 0

    return {
        'overall_feedback': response,
        'average_score': average_score
    }

# Define Graph
graph = StateGraph(UPSCState)

# Add nodes to the graph
graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_clarity', evaluate_clarity)
graph.add_node('final_evaluation', final_evaluation)


# Define edges (parallel execution)
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_clarity')
graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_clarity', 'final_evaluation')
graph.add_edge('final_evaluation', END)


# Compile the graph
workflow = graph.compile()

# Execute the workflow
initial_state = {
    "essay": essay
}
final_state = workflow.invoke(initial_state)
print("\nOverall Feedback:\n", final_state['overall_feedback'])
print("\nAverage Score:", final_state['average_score'])

# Print the graph visualization
# print(workflow.get_graph().draw_mermaid())