import os

from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
load_dotenv()

os.environ["OPENAI_API_KEY"]

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")


# Define prompt
prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
llm_chain = LLMChain(llm=llm, prompt=prompt)

chain = load_summarize_chain(llm, chain_type="refine" , document_variable_name="text")

docs = loader.load()
print(chain.run(docs))


