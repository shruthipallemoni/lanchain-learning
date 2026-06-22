from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import Language
# from langchain_community.document_loaders import PyPDFLoader

text="""
   from django.apps import AppConfig


class UsersConfig(AppConfig):
    name = "users"

    def ready(self):
        import users.signals 

"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0
)
chunks=splitter.split_text(text)
print(chunks)
# print(chunks[0].page_content)