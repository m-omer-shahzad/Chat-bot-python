import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
os.environ["OPENAI_API_KEY"]
llm = OpenAI(temperature=0)
   
class outputParser(BaseModel):
    is_greeting: bool = Field(description="if the output is yes then return True else return False")

    def to_dic(self):
        return  self.is_greeting

parser =PydanticOutputParser(pydantic_object=outputParser)

prompt = PromptTemplate(
    template="\n{format_instructions}\nHere is a message:{query}Is this is a greating message?",
    input_variables=["query"],
    partial_variables={"format_instructions":  parser.get_format_instructions()}
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

while True:
    query = input("Enter Question:")
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        break
    if query == '':
        continue
    answer = parser.parse(llm_chain.run(query)).to_dic()
    print(answer)
