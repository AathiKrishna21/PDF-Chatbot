from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document
from typing import List
from langchain_core.embeddings import Embeddings

load_dotenv()


def setup_chroma_db(embeddings: Embeddings, docs: List[Document]) -> Chroma:
    """Setup Chroma DB with the specified embeddings."""
    db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="emb"
    )
    return db


def search_facts(db: Chroma, query: str):
    """Search for facts in the Chroma DB."""
    results = db.similarity_search(query)
    for result in results:
        print("\n")
        print("Fetched Context: ", result.page_content)
    return results


if __name__ == "__main__":
    search_facts("What is an interesting fact about the English language?")
