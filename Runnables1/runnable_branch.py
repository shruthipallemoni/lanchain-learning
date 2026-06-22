from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()
from langchain.schema.runnable import RunnablePassthrough,RunnableParallel,RunnableSequence,RunnableBranch

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)

prompt1=PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Summarixe the following texy \n{text}",
    input_variables=['text']
)

parser=StrOutputParser()

rep_chain=RunnableSequence(prompt1,model,parser)

branch_chain=RunnableBranch(
    (lambda x:len(x.split())>500,RunnableSequence(prompt2,model,parser)),
    RunnablePassthrough()
)


res=RunnableSequence(rep_chain,branch_chain)
print(res.invoke({'topic':'black-hole'}))