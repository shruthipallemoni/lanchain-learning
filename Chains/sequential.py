from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)

#1st prompt
template1=PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)


#2nd prompt
template2=PromptTemplate(
    template='Write a 5 line summary on the following text./n {text}',
    input_variables=['text']
)
#with using Output Parser

parser=StrOutputParser()

chain=template1|model|parser|template2|model|parser

res=chain.invoke({'topic':'black hole'})
print(res)
chain.get_graph().print_ascii()