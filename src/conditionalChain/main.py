import os

import langchain
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
def detect_salutation(original_message) -> bool:
    prompt = PromptTemplate(
        template="\n{format_instructions}\nHere is a message:{query}Is this is a greating message?",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return parser.parse(llm_chain.run(original_message)).to_dic()

# Revised message output parser
class revisedMessageParser(BaseModel):
    revised_message: str = Field(
        description="When the output is revised and any connotations are eliminated."
    )

    def to_dict(self):
        return {"Revised_Message": self.revised_message}

revisedMessageParsers = PydanticOutputParser(pydantic_object=revisedMessageParser)

# Prompt to revise message
def revise_chain(original_message):
    prompt = PromptTemplate(
        template="""
            Prompt: Write a conversational message that revises the given text. Make the revised message sound like a natural conversation, avoiding the use of email-like elements such as greetings, sender's name, and thank-you notes. Ensure that the revised message doesn't contain negative connotations.
            FormateInstructions: {format_instructions}
            Original Message: {original_message}""",
        input_variables=["original_message"],
        partial_variables={
            "format_instructions": revisedMessageParsers.get_format_instructions()
        },
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return revisedMessageParsers.parse(llm_chain.run(original_message)).to_dict()

# connotation output parser
class connotationParser(BaseModel):
    connotation: list = Field(
        description="The output in the array of objects in which there are three keys of connotation , reason and impact."
    )

    def to_dic(self):
        return {"connotations": self.connotation}

connotationParsers = PydanticOutputParser(pydantic_object=connotationParser)

# prompt to list connotation
def connotation_chain(original_message):
    prompt = PromptTemplate(
        template="""
            Prompt: Please disregard any prior instructions and approach this task anew.
            Examine the provided message: {original_message}. Identify any underlying connotations within the message, and provide an explanation for each connotation you identify. Additionally, elucidate how each connotation could potentially influence the recipient. Organize your response using an array of objects. Each object should encompass a title, the reason behind its recognition, and the potential impact on the receiver. Please maintain a succinct and focused response. Do not include connotations with positive sentiment.
            FormateInstructions: {format_instructions}
            """,
        input_variables=["original_message"],
        partial_variables={
            "format_instructions": connotationParsers.get_format_instructions()
        },
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return connotationParsers.parse(llm_chain.run(original_message)).to_dic()

# Main function
def main():
    original_message = input("Enter Message: ")
    answer = detect_salutation(original_message)
    if answer:
        print("original_Message: ", original_message)
    else:
        print(revise_chain(original_message))
        print(connotation_chain(original_message))

langchain.debug= True
if __name__ == "__main__":
    main()

