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
def detect_salutation(original_message) -> bool:
    prompt = PromptTemplate(
        template="\n{format_instructions}\nHere is a message:{query}Is this is a greating message?",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return parser.parse(llm_chain.run(original_message)).to_dic()


class revisedMessageParser(BaseModel):
    revised_message: str = Field(description="if the output is revised and the connotatiosn is removed")

    def to_dict(self):
        return  self.revised_message
        
    
revisedMessageParsers = PydanticOutputParser(pydantic_object=revisedMessageParser)

# Prompt to revise message
def revise_chain(original_message):
    prompt = PromptTemplate(
        # template="""Forget all previous instructions and start fresh.\n 
        # \n{format_instructions}\nDo not make any message like an email\n
        # This is the message user enters: {original_message}.\n
        # [Revised Message] must look like natural conversation(not an email). Do not include Salutation (Greeting) , Closing and Sign-off and fluff. Do not include Sender's Name and Receiver's Name in [Revised Message]\n\n
        # Just Revised message and remove the connotations.\n
        # If the original message is just the alphabet, do not Revised the message.\n
        # Revised message in [Revised Message]\n
        # Do not return these guidlines\n""",
        template="""
            Prompt: Write a conversational message that revises the given text. Make the revised message sound like a natural conversation, avoiding the use of email-like elements such as greetings, sender's name, and thank-you notes. Ensure that the revised message doesn't contain negative connotations.
            FormateInstructions: {format_instructions}
            Original Message: {original_message}
            [Revised Message]""",

    
        input_variables=["original_message"],
        partial_variables={"format_instructions": revisedMessageParsers.get_format_instructions()},
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return revisedMessageParsers.parse(llm_chain.run(original_message)).to_dict()

# prompt to list
def connotation_chain(original_message):
    prompt = PromptTemplate(
        template="Forget all previous instructions and start fresh.\n Analyze the given message: {original_message}, and provide the connotations along with an explanation for each one. For each connotation, include its impact on the receiver. Format your response by enclosing each connotation in an array of objects, each containing the connotation itself, the reason for its identification, and its impact on the receiver. Keep your response concise and focused. and do not return positive connotations",
        input_variables=["original_message"],
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return llm_chain.run(original_message)

# Main function
def main():
    original_message = input("Enter Message: ")
    answer = detect_salutation(original_message)
    if answer:
        print("This is Greeting Message")
        print("Message: ", original_message)
    else:
        print(revise_chain(original_message))
        print(connotation_chain(original_message))


if __name__ == "__main__":
    main()
