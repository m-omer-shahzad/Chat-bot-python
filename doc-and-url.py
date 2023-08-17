import os
import sys
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import (
    CSVLoader, Docx2txtLoader, PyPDFLoader, TextLoader)
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from colors import green, white, yellow
from langchain.document_loaders import UnstructuredURLLoader

load_dotenv()
os.environ["OPENAI_API_KEY"]


def load_documents_from_files():
    documents = []

    # Dictionary mapping file extensions to corresponding loader classes
    loaders = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.doc': Docx2txtLoader,
        '.txt': TextLoader,
        '.csv': CSVLoader
    }

    # Loop through the files in the "docs" directory
    for file in os.listdir("docs"):
        extension = os.path.splitext(file)
        if extension in loaders:
            loader = loaders[extension]
            pdf_path = os.path.join("docs", file)
            doc_loader = loader(pdf_path)
            documents.extend(doc_loader.load())

    return documents


def create_vector_space(documents, space_directory):
    # Split documents into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    # Create a Chroma vector database from the documents using OpenAI embeddings
    vectordb = Chroma.from_documents(
        documents, embedding=OpenAIEmbeddings(), persist_directory=space_directory)
    vectordb.persist()

    return vectordb


def print_interaction_status(interaction_type):
    if interaction_type:
        print(
            f'You are interacting with {interaction_type}. Type your question or enter "exit" to return')
    else:
        print('You are now in the main menu. Please select an option.')


def main():
    interaction_type = None  # To track the interaction type
    url_space_directory = "./url_data"
    doc_space_directory = "./doc_data"
    print(f"\n{yellow}-------------------------")
    print('Welcome to the DocBot!')
    print("-------------------------")
    while True:
        print('\nSelect an option:')
        print('1. Interact with provided URLs')
        print('2. Interact with documents')
        print('q. Exit the program')

        option = input('\nEnter your choice: ')

        if option == '1':
            interaction_type = 'URLs'
            urls = ["https://omershahzad70.blogspot.com/"]
            loader = UnstructuredURLLoader(urls=urls)
            documents = loader.load()
            url_vectordb = create_vector_space(documents, url_space_directory)
            doc_vectordb = None

        elif option == '2':
            interaction_type = 'documents'
            documents = load_documents_from_files()
            doc_vectordb = create_vector_space(documents, doc_space_directory)
            url_vectordb = None

        elif option.lower() == 'q':
            print('\nExiting')
            sys.exit()
        else:
            print('\nInvalid option. Please choose again.')
            continue

        qa_bot = None
        if interaction_type == 'URLs':
            retriever = url_vectordb.as_retriever(search_kwargs={'k': 1})
        elif interaction_type == 'documents':
            retriever = doc_vectordb.as_retriever(search_kwargs={'k': 1})

        if retriever:
            qa_bot = RetrievalQA.from_chain_type(
                llm=OpenAI(),
                retriever=retriever,
                return_source_documents=True
            )

        print(
            f"\n{yellow}---------------------------------------------------------------------------------")
        print_interaction_status(interaction_type)
        print('---------------------------------------------------------------------------------')

        while True:
            query = input(f"{green}Question: ")
            if query.lower() in ["exit", "quit", "q"]:
                print('Returning to main menu')
                break
            if query == '':
                continue
            if qa_bot:
                result = qa_bot({'query': query})
                print(f"{white}Answer: " + result["result"])
            else:
                print('Please select an option first.')


if __name__ == "__main__":
    main()
