from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)

chat_template=ChatPromptTemplate(
    [
        ('system','You are helpful {domain} expert'),
        ('human','Explain in simple terms ,what is {topic}')
    ]
)
 
chain=chat_template|model

result=chain.invoke({
    'domain':'cricket',
    'topic':'Pitch'})

print("Result : ",result)
# print(prompt)