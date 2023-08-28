import os

import langchain
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from salutation_Detector import salutationChain, parser as SalutationParser


def initialize_llm():
    load_dotenv()
    os.environ["OPENAI_API_KEY"]
    return ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")


llm = initialize_llm()

final_response = {
    "original_message": None,
    "is_salutation": False,
    "is_revised": True,
    "revised_message": True,
    "connotations": None,
}


def detect_salutation(original_message):

    salutation_detector_chain = salutationChain().chain()
    final_response["is_salutation"] = SalutationParser.parse(
        salutation_detector_chain.run(message=original_message)
    ).to_dic()
    final_response["is_revised"] = False

# Revised message output parser
class outputParser(BaseModel):
    revised_message: str = Field(
        description="When the output is revised and any connotations are eliminated."
    )
    connotation: list = Field(
        description="The output in the array of objects in which there are three keys of connotation , reason and impact."
    )

    def to_dict(self):
        return {
            "revised_message": self.revised_message,
            "connotations": self.connotation,
        }


parser = PydanticOutputParser(pydantic_object=outputParser)


# Prompt to revise message
def revise_chain(original_message):
    prompt = PromptTemplate(
        template="""
            Prompt: Write a conversational message that revises the given text. 
            Make the revised message sound like a natural conversation, avoiding the use of email-like elements such as greetings, sender's name, and thank-you notes. Ensure that the revised message doesn't contain negative connotations.Please disregard any prior instructions and approach this task anew.
            Identify any underlying connotations within the message, and provide an explanation for each connotation you identify. Additionally, elucidate how each connotation could potentially influence the recipient. Organize your response using an array of objects. Each object should encompass a title, the reason behind its recognition, and the potential impact on the receiver. Please maintain a succinct and focused response. Do not include connotations with positive sentiment.

            FormateInstructions: {format_instructions}
            Original Message: {original_message}""",
        input_variables=["original_message"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return parser.parse(llm_chain.run(original_message)).to_dict()


# Main function
def main():
    original_message = input("Enter Message: ")
    final_response["original_message"] = original_message

    if final_response["is_salutation"]:
        print(final_response)
    else:
        output = revise_chain(original_message)
        final_response["revised_message"] = output['revised_message']
        final_response["connotations"] = output['connotations']

        print(final_response)


# langchain.debug= True
if __name__ == "__main__":
    main()
