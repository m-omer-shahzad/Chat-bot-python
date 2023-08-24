import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
os.environ["OPENAI_API_KEY"]

llm = OpenAI(temperature=0)

prompt = PromptTemplate(
    template="Here is a given message:{query}Must include list of Identified Cannotations, enclose each Connotation in array of objects, also include reason explaining why Connotation was identified and impact on receiver against each connotation in your response.\nn",
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
