import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
os.environ["OPENAI_API_KEY"]

llm = OpenAI(temperature=0)

prompt = PromptTemplate(
    template="Here is a message:{query}Is this is a greating message if yes then return yes otherwise no?",
    input_variables=["query"]
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

while True:
    query = input("Enter Question:")
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        break
    if query == '':
        continue
    answer = llm_chain.run(query)
    print(answer)
