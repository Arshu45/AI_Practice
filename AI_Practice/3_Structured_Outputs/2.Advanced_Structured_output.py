from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Annotated, Optional
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Schema
class Review(BaseModel):
    key_themes: List[str] = Field(description="Write top 3 Key themes discussed in the review")
    summary: str = Field(description="Summary of the review")
    sentiment: str = Field(description="Sentiment of the review, either positive, negative, or neutral")
    pros: Optional[List[str]] = Field(default=None, description="List of pros mentioned in the review")
    cons: Optional[List[str]] = Field(default=None, description="List of cons mentioned in the review")

structured_model = model.with_structured_output(Review)

prompt = """I’ve been using the Google Pixel for a few weeks now, and overall it’s been a very satisfying experience. The camera quality is outstanding — photos come out sharp with excellent color accuracy, even in low-light conditions. The clean Android experience is another big plus, with no unnecessary apps and fast software updates directly from Google.

Performance-wise, the phone feels smooth for daily tasks like browsing, social media, and streaming. Battery life is decent and usually lasts me a full day with moderate usage, though heavy camera or gaming use can drain it faster. The display is bright and clear, making videos and reading very enjoyable.

On the downside, the phone can get slightly warm during extended use, and charging speed is slower compared to some competitors in the same price range. Also, the hardware design is simple and may not stand out if you prefer a more premium or flashy look.

Overall, the Google Pixel is a great choice if you value camera quality, clean software, and long-term updates. I’d recommend it to anyone looking for a reliable Android phone with an excellent photography experience."""

response = structured_model.invoke(prompt)
print(response)  
