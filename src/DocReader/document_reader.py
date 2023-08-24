import os
import sys
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from colors import green, white, yellow

load_dotenv()

os.environ["OPENAI_API_KEY"]

documents = []


# Function to load a document using a specific loader
def generateDoc(file, loaderType):
    pdf_path = "./docs/" + file
    loader = loaderType(pdf_path)
    documents.extend(loader.load())


# Dictionary mapping file extensions to corresponding loader classes
loaders = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
}

# Loop through the files in the directory
for file in os.listdir("docs"):
    filename, extension = os.path.splitext(file)
    if extension in loaders:
        loader = loaders[extension]
        generateDoc(file, loader)

# Split documents into smaller chunks for processing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Create a Chroma vector database from the documents using OpenAI embeddings
vectordb = Chroma.from_documents(
    documents, embedding=OpenAIEmbeddings(), persist_directory="./cache/data"
)
vectordb.persist()

qa_bot = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever(search_kwargs={"k": 1}),
    return_source_documents=True,
)

print(
    f"{yellow}---------------------------------------------------------------------------------"
)
print(
    "Welcome to the DocBot. You are now ready to start interacting with your documents"
)
print(
    "---------------------------------------------------------------------------------"
)
while True:
    query = input(f"{green}Question: ")
    if query == "exit" or query == "quit" or query == "q":
        print("Exiting")
        sys.exit()
    if query == "":
        continue
    result = qa_bot({"query": query})
    print(f"{white}Answer: " + result["result"])
