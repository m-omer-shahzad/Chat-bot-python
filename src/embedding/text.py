import getpass
import os

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

load_dotenv()
os.environ["OPENAI_API_KEY"]
MONGODB_ATLAS_CLUSTER_URI  = os.environ["MONGODB_ATLAS_CLUSTER_URI"]


loader = TextLoader("./docs/state_of_the_union.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()


def initialize_mongo_client():
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
    return docsearch


# perform a similarity search between the embedding of the query and the embeddings of the documents
query = "What did the president say about Ketanji Brown Jackson"
docSearch = initialize_mongo_client()
docs = docSearch.similarity_search(query)
print(docs[0].page_content)