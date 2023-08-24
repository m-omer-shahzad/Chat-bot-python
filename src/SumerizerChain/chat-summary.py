import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document

def summarize_conversation(conversation):
    load_dotenv()
    os.environ["OPENAI_API_KEY"]

    docs = [Document(page_content=conversation)]
    model = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
    summary_chain = load_summarize_chain(model, chain_type="refine", verbose=True)

    return summary_chain.run(docs)

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

summary_result = summarize_conversation(conversation)
print(summary_result)
