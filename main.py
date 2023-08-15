import sys
import os
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader

from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]

documents = []

def generateDoc(file, loaderType):
    pdf_path = "./docs/" + file
    loader = loaderType(pdf_path)
    documents.extend(loader.load())
    
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        generateDoc(file, PyPDFLoader)
    elif file.endswith('.docx') or file.endswith('.doc'):
        generateDoc(file, Docx2txtLoader)
    elif file.endswith('.txt'):
        generateDoc(file, TextLoader)
    elif file.endswith('.csv'):
        generateDoc(file, CSVLoader)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

vectordb = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory="./data"  )
vectordb.persist()

qa_bot = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever(search_kwargs={'k': 1}),
    return_source_documents=True
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Question: ")
    if query == "exit" or query == "quit" or query == "q" or query == "f":
        print('Exiting')
        sys.exit()
    if query == '':
        continue
    result = qa_bot({'query': query})
    print(f"{white}Answer: " + result["result"])
    chat_history.append((query, result["result"]))
