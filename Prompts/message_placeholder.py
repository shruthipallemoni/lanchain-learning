from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
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

#chat template
chat_template=ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
     MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]

#load chat history
with open('chat_history.py') as f:
    chat_history.extend(f.readlines())

print(chat_history)

#create prompt
prompt=chat_template.invoke({
    'chat_history':chat_history,
    'query':"Where is my refund"
})
print(prompt)

#Call the Model
result=model.invoke(prompt)
print(result)