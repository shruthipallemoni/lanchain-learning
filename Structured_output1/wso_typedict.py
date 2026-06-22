from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from typing import TypedDict,Annotated,List,Literal,Optional
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)

class Review(TypedDict):
    key_themes:Annotated[List[str],"Return the key points from the summary"]
    summary:Annotated[str,"Return the brief summary of the Review"]
    sentiment:Annotated[Literal["pos","neg"],"Return the sentiment of the Review either postive or negative or neutral"]
    cons:Annotated[Optional[list[str]],"Return cons from the Review"]
    pros:Annotated[Optional[list[str]],"Return pros from the Review"]
    name:Annotated[Optional[str],"Return the name of the Reviewer"]

structured_model=model.with_structured_output(Review)

# result=structured_model.invoke("""The hardware is great but the software is bloated,There are many preinstalled apps that I can't remove.Also the UI looks so outdated..""")

result=structured_model.invoke("""The hardware is great but the software is bloated,There are many preinstalled apps that I can't remove.Also the UI looks so outdated..""")
print(result)
print(result['summary'])
print(result['sentiment'])