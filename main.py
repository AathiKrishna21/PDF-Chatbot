import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from embeddings import create_embeddings
from vector_store import setup_chroma_db
from chain import build_chain


def read_pdf(pdf_file):
    print("pdf: ", pdf_file)
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,
        chunk_overlap=0
    )
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load_and_split(
        text_splitter=text_splitter
    )
    return docs


def process_input(input, db, chat_history):
    chain = build_chain(db)
    response = chain.invoke({
        "question": input,
        "chat_history": chat_history
    })
    print(response.keys())
    print(response["answer"])
    print(type(response["answer"]))
    if isinstance(response["answer"], dict):
        print(response["answer"].keys())
    ans = {"role": "ai", "content": response["answer"]}
    st.session_state.messages.append(ans)
    with st.chat_message("ai"):
        st.markdown(ans["content"])


def start_ui():
    db = None
    embeddings = create_embeddings(mode="local")
    st.set_page_config(page_title="PDF Chatbot")
    st.header("RAG-based LLM for PDF Analysis", divider='rainbow')

    if "messages" not in st.session_state:
        init_messages = [
            {
                "role": "ai",
                "content": "Hello! Good to See You. "
                "Malini this side, May I Know how to help you!"
            }
        ]
        st.session_state.messages = init_messages

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Retrieve db from session_state if available
    db = st.session_state.get("db", None)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files Here",
            accept_multiple_files=False
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                docs = read_pdf(pdf_docs)
                # text_chunks = split_text_chunks(raw_text)
                db = setup_chroma_db(embeddings, docs)
                st.session_state["db"] = db  # Save db to session_state
                st.success("Done")

    if promt := st.chat_input("Ask me anything from the Document."):
        with st.chat_message("human"):
            st.markdown(promt)
        st.session_state.messages.append({"role": "human", "content": promt})
        chat_history = [
            (msg["role"], msg["content"])
            for msg in st.session_state.messages
            if msg["role"] in ("human", "ai")
        ]
        process_input(promt, db, chat_history)


def main():
    start_ui()


if __name__ == "__main__":
    main()
