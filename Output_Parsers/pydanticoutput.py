from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)

class Person(BaseModel):
    name:str=Field(description="Name of the person")
    age:int=Field(gt=18,description="Enter the age > 18")
    city:str=Field(description="City name of the person belongs to")

parser=PydanticOutputParser(Person)

template=PromptTemplate(
    template="Generate the name,age and city of a fictional {place} person.\n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt=template.invoke({'topic':'indian'})
# result=model.invoke(prompt)
# final_result=parser.parse(result.content)
# print(final_result)

chain=template|model|parser

res=chain.invoke({'place':'indian'})
print(res)