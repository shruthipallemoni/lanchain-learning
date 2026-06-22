from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import pydantic
from pydantic import Field,list
from typing import Literal,Optional
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)

#schema
json_schema={

}
structured_model=model.with_structured_output(json_schema)


result=structured_model.invoke("""The hardware is great but the software is bloated,There are many preinstalled apps that I can't remove.Also the UI looks so outdated..""")
print(result)
print(result['summary'])
print(result['sentiment'])