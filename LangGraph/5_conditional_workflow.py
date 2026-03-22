# LLM Based review handling conditional workflow

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os

load_dotenv()

# Safety check (important)
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("GOOGLE_API_KEY not found in environment")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(..., description="The sentiment of the review, either 'positive' or 'negative'.")

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')

structured_model = model.with_structured_output(SentimentSchema)
structured_model_2 = model.with_structured_output(DiagnosisSchema)

# prompt = "Determine the sentiment of the following review: 'The product quality is excellent and I am very satisfied with my purchase.'"

# response = structured_model.invoke(prompt)
# print("Sentiment:", response.sentiment)

# Define State
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str

# Define functions to call LLM
def find_sentiment(state: ReviewState) -> ReviewState:
    review = state["review"]

    prompt = f"For the following review determine the sentiment (positive or negative):\n\n{review}"

    response = structured_model.invoke(prompt)
    return {'sentiment': response.sentiment}


def positive_response(state: ReviewState) -> ReviewState:
    review = state["review"]

    prompt = f"Write a warm thankyou message in response to this review:\n\n{review}, Also kindly ask the user to leave feedback on the website."

    response = model.invoke(prompt)
    return {'response': response.content}

def run_diagnosis(state: ReviewState) -> ReviewState:
    review = state["review"]

    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency.
"""
    response = structured_model_2.invoke(prompt)
    return {'diagnosis': response.model_dump()}

def negative_response(state: ReviewState) -> ReviewState:
    diagnosis = state['diagnosis']

    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = model.invoke(prompt).content

    return {'response': response}

def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"



graph = StateGraph(ReviewState)

graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('negative_response', negative_response)


graph.add_edge(START, 'find_sentiment')
graph.add_conditional_edges('find_sentiment', check_sentiment)
graph.add_edge('positive_response', END)
graph.add_edge('run_diagnosis', 'negative_response')
graph.add_edge('negative_response', END)
graph.add_edge('find_sentiment', END)

workflow = graph.compile()


# Visualize the graph
# print(workflow.get_graph().draw_mermaid())

# Example usage
initial_state = {
    'review': "The product broke after one week of use. Very disappointed and a waste of money."
}

response = workflow.invoke(initial_state)
print("Final State:", response)

