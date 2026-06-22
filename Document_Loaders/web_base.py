from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0,
    api_key=api_key
)

parser=StrOutputParser()
url='https://www.geeksforgeeks.org/problem-of-the-day'
loader=WebBaseLoader(
    url
)

docs=loader.load()
print(docs)
print(len(docs))
print(type(docs))
print(docs[0].page_content)
prompt=PromptTemplate(
    template='Answer the following  \n{question} from the given text \n {text}',
    input_variables=['question','text']
)
chain=prompt|model|parser
print(chain.invoke({'question':'What is the Problem of the day title?','text':docs[0].page_content}))

