import os

from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

load_dotenv()

os.environ["OPENAI_API_KEY"]
model = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")


template = """
    Write a concise summary of the conversation between Jim and Pam from the TV show "The Office":
    {text} do noreturn the original text"""

conversation = """
    Jim: I can’t believe we got lost in Paris, of all places.
    Pam: I know, right? But I have to admit, it’s kind of romantic wandering around these streets with you.
    Jim: Yeah, it’s like we’re in our own little movie.”
    Pam: Speaking of movies, we should find a cute little cafe and have some croissants and coffee.
    Jim: Sounds perfect. But first, let’s take a selfie in front of the Eiffel Tower.
    Pam: Yes! And then we can send it to Dwight and Michael to make them jealous.
    Jim: Ha! They’re probably struggling to order food in French right now.
    Pam: Well, at least we have each other to navigate this city with.
    Jim: Always, Pam. Always.
    """

docs = [Document(page_content=conversation)]
prompt = PromptTemplate(template=template, input_variables=["text"])
summary_chain = load_summarize_chain( model, chain_type="stuff", verbose=True, prompt=prompt)


summary_result = summary_chain.run(docs)
print(summary_result)
