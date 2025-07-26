from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
import os


def get_model() -> ChatOpenAI:
    """Get the chat model based on the environment variable."""
    # Initialize the chat model
    mode = os.environ.get("MODEL_MODE", "local")
    if mode == "local":
        llm = os.environ.get("LLM_MODEL_NAME")
        base_url = os.environ.get("LLM_BASE_URL")
        return ChatOpenAI(
            model_name=llm,
            openai_api_base=base_url,
            openai_api_key="",
        )
    elif mode == "remote":
        # API key in .env is mandatory
        return ChatOpenAI()
    else:
        raise ("MODEL_MODE environment variable should be local/remote")


def build_chain(db: Chroma) -> ConversationalRetrievalChain:
    """
    Build a conversational retrieval chain 
    using the provided embeddings and database.
    """
    retriever = db.as_retriever()

    chain = ConversationalRetrievalChain.from_llm(
        llm=get_model(),
        retriever=retriever,
    )

    return chain
