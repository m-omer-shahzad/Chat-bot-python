import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.document_loaders import JSONLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

load_dotenv()

MONGODB_ATLAS_CLUSTER_URI  = os.environ["MONGODB_ATLAS_CLUSTER_URI"]

os.environ["OPENAI_API_KEY"]



file_path='./docs/data.json'
data = json.loads(Path(file_path).read_text())
print("data", data)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(data)

print(docs , "desddsd")
embeddings = OpenAIEmbeddings()


# initialize MongoDB python client
client = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

db_name = "health-reel"
collection_name = "health-reel"
collection = client[db_name][collection_name]
index_name = "health-reel-index"

# insert the documents in MongoDB Atlas with their embedding
docsearch = MongoDBAtlasVectorSearch.from_documents(
    docs, embeddings, collection=collection, index_name=index_name
)


# perform a similarity search between the embedding of the query and the embeddings of the documents
query = "what is my average heartRate"
docs = docsearch.similarity_search(query)

print(docs , "docss")
print(docs[0].page_content)


