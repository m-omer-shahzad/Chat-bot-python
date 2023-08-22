from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
import os
import langchain

from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"]


llm = OpenAI(temperature=0.7)


template = "This is the messahe user enter {original_message} do not change the context of the message just revize this message and remove the connotations"

prompt_template = PromptTemplate(input_variables=["original_message"], template=template)
revise_message_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="revise_message")

template = (
    "identify the connotations from this message: {revise_message}, and list down all connotation then enclose each Connotation in array of objects, also include reason explaining why Connotation was identified and impact on receiver against each connotation in your response.\nn do not wtite other stuff other then this"
)
prompt_template = PromptTemplate(input_variables=["revise_message"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="connotation")


overall_chain = SequentialChain(
    chains=[revise_message_chain, review_chain],
    input_variables=["original_message"],
    output_variables=["revise_message", "connotation" , "original_message"],
    verbose=True,
)

langchain.debug = True
original_message = input("Enter message: ")
print(overall_chain({"original_message": original_message}))
