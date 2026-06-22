from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
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

messages=[
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Tell me about LangChain")
]

result=model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)