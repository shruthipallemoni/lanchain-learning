from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

api_key = os.getenv("GROQ_API_KEY")
load_dotenv()
model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)
res=model.invoke("What is the capital of India?")
print(res)
print(res.content)