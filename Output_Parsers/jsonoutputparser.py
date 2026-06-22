from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)
#This model won't give with_structured_output function
#so we are using output parsers here

parser=JsonOutputParser()
template=PromptTemplate(
    template='Give me the name,age and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
) 

chain=template|model|parser

result=chain.invoke({})

print(result)

#The disadvantage is that
#we can't enforce schema
#which means we can't get the result in the stuctured format