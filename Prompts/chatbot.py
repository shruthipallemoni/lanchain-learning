from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)

chat_history=[
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input=input('You :')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI :",result.content)

print(chat_history) 