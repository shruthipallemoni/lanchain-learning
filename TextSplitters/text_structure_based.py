from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader(
    'pdfs/LLM.pdf'
)
docs=loader.load()

splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
chunks=splitter.split_documents(docs)
print(chunks)
print(chunks[0].page_content)