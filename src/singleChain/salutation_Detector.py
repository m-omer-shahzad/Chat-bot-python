import os


from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

def initialize_llm():
    load_dotenv()
    os.environ["OPENAI_API_KEY"]
    return ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")

llm = initialize_llm()

# salutations output parser
class salutationParser(BaseModel):
    is_greeting: bool = Field(
        description="If the output is yes then return True else return False"
    )

    def to_dic(self):
        return self.is_greeting

parser = PydanticOutputParser(pydantic_object=salutationParser)

# Prompt to detect salutation
class salutationChain:

    template="\n{format_instructions}\nHere is a message:{message}. Is this is a greating message?"
    input_variables=["message"]

        
    def chain(self):
        prompt_template = PromptTemplate(
            input_variables= self.input_variables,
            template=self.template,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        salutationResponse = LLMChain(llm=llm, prompt=prompt_template)
        return  salutationResponse