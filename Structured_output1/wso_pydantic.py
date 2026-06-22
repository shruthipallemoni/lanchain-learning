from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

from pydantic import BaseModel, Field
from typing import Literal, Optional

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)

class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )
    summary: str = Field(
        description="A brief summary of the review"
    )
    sentiment: Literal["pos", "neg", "neutral"] = Field(
        description="Return the sentiment of the review"
    )
    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros in the list"
    )
    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons in the list"
    )
    name: Optional[str] = Field(
        default=None,
        description="Write down the name of the reviewer"
    )

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """The hardware is great but the software is bloated.
    There are many preinstalled apps that I can't remove.
    Also the UI looks so outdated."""
)

print(result)
print(result.summary)
print(result.sentiment)