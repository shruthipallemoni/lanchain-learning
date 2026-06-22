from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader,PyPDFLoader
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
loader=PyPDFLoader("pdfs/LLM.pdf")

docs=loader.load()
print(docs)
print(len(docs))
print(type(docs))
print(docs[0].page_content)
prompt=PromptTemplate(
    template='Write a breif summary of the content \n{content}',
    input_variables=['content']
)
chain=prompt|model|parser
# print(chain.invoke({'content':docs[0].page_content}))

