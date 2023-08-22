import os

import json
from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def initialize_llm():
    load_dotenv()
    os.environ["OPENAI_API_KEY"]
    llm = OpenAI(temperature=0.0)
    return llm

def revise_chain(llm):
    template = "This is the message user enters: {original_message} Do not change the context of the message. Just revise this message and remove the connotations. If this is the greeting message, do not revise the original message make the field empty. If the message is just the alphabet, do not revise the message. Do not include fluff in your response. dont add irrelevant space"

    promptTemplate = PromptTemplate(input_variables=["original_message"], template=template)
    return LLMChain(llm=llm, prompt=promptTemplate, output_key="revise_message")

def connotation_chain(llm):

    template = "Analyze the given message: {original_message}. Identify any connotation words present in the message, and provide a list of these connotations along with an explanation for each one. For each connotation, include its impact on the receiver. Format your response by enclosing each connotation in an array of objects, each containing the connotation itself, the reason for its identification, and its impact on the receiver. Keep your response concise and focused."
       
    promptTemplate = PromptTemplate(input_variables=["original_message"], template=template)
    return LLMChain(llm=llm, prompt=promptTemplate, output_key="connotation")

   

def sequential_chain(revise_chain, connotation_chain):
    return SequentialChain(
        chains=[revise_chain, connotation_chain],
        input_variables=["original_message"],
        output_variables=["original_message" , "revise_message", "connotation"],
        verbose=True,
    )

def main():
    llm = initialize_llm()

    reviseChain = revise_chain(llm)
    connotationChain = connotation_chain(llm)

    sequentialChain = sequential_chain(reviseChain, connotationChain)
    original_message = input("Enter message: ")
    
    output_data = sequentialChain({"original_message": original_message})
    print(output_data)  
    
    with open("resdata.json", "w") as outfile:
        json.dump(output_data, outfile)

if __name__ == "__main__":
    main()
