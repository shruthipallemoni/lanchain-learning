from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel  
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model1= ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)
model2=()

prompt1=PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=['text']
)

prompt2=PromptTemplate(
    template="Generate 5 short question answers from the following text.\n {text}",
    input_variables=['text']
)

prompt3=PromptTemplate(
    template="Merge the provided notes and quiz into a single document. notes->{notes} and quiz -> {quiz}",
    input_variables=['notes','quiz']
)

#with using Output Parser
parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1|model1|parser,
    'quiz':prompt2|model2|parser
})
merge_chain=prompt3|model1|parser

chain=parallel_chain|merge_chain

text="""
   
"""

res=chain.invoke({'text':text})
print(res)
chain.get_graph().print_ascii()