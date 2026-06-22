from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader

import os
from dotenv import load_dotenv

load_dotenv()

loader=CSVLoader(
   'sample_data/sample.csv',
)

docs=loader.load()
print(docs)
print(len(docs))
print(type(docs))
print(docs[0].page_content)

