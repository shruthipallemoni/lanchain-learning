from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from typing import TypedDict,Annotated,List,Literal,Optional
load_dotenv()
from langchain.schema.runnable import RunnablePassthrough,RunnableParallel,RunnableSequence,RunnableLambda

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)

def word_counter(text):
    return len(text.split())

prompt=PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

parser=StrOutputParser()
joke_gen_chain=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'count_words':RunnableLambda(word_counter)
})

res=RunnableSequence(joke_gen_chain,parallel_chain)
print(res.invoke({'topic':'black-hole'}))