from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv
from typing import List
import requests

load_dotenv()


class LLMSCustomEmbeddings(Embeddings):
    def __init__(
        self,
        endpoint_url: str = "http://10.2.0.2:3333/v1/embeddings",
        model: str = "text-embedding-bge-base-en-v1.5"
    ):
        self.endpoint_url = endpoint_url
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {
            "model": self.model,
            "input": texts
        }
        response = requests.post(self.endpoint_url, json=payload)
        response.raise_for_status()
        # Return embeddings for all documents
        return [item["embedding"] for item in response.json()["data"]]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def create_embeddings(mode: str = "local") -> Embeddings:
    """Create embeddings based on the specified mode."""
    if mode == "local":
        return LLMSCustomEmbeddings()
    elif mode == "remote":
        return OpenAIEmbeddings()
    else:
        raise ValueError("Invalid mode. Choose 'local' or 'remote'.")