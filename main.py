import streamlit as st
import asyncio
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain_huggingface  import HuggingFacePipeline
from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from langchain_google_genai import GoogleGenerativeAIEmbeddings


# class SentenceTransformersEmbeddings(Embeddings):
#     def __init__(self, model_name="all-MiniLM-L6-v2"):
#         # Initialize the Sentence-Transformer model
#         self.model = SentenceTransformer(model_name)
    
#     def embed_documents(self, texts):
#         # Encode a list of texts to generate their embeddings
#         return [self.model.encode(text) for text in texts]
    
#     def embed_query(self, text):
#         # Encode a single text query to generate its embedding
#         return self.model.encode(text)


def read_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = [model.encode(chunk) for chunk in text_chunks]
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model_name = "gpt2"  # You can replace this with a more powerful model like 'gpt-neo', 'gpt-j', or any other open-source GPT model
    generator = pipeline("text-generation", model=model_name, max_length=6144, max_new_tokens=6144,)
    # generator.model.config.max_length = 1024  # Increase max length for the input text
    # generator.model.config.max_new_tokens = 200
    # Wrap the pipeline into Langchain's HuggingFacePipeline
    hf_pipeline = HuggingFacePipeline(pipeline=generator)
    # model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=hf_pipeline, prompt=prompt)
    return chain

def process_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    context = "\n".join([doc.page_content for doc in docs])
    chain = get_conversational_chain()
    response = chain.invoke({"context": context, "question": user_question})
    print(response.keys())
    print(response["text"])
    print(type(response["text"]))
    if isinstance(response["text"], dict):
        print(response["text"].keys())
    ans = {"role": "ai", "content": response["text"]}
    st.session_state.messages.append(ans)
    with st.chat_message("ai"):
        st.markdown(ans["content"])
    # st.write("Reply: ", response["Reply"])

genai.configure(api_key=st.secrets["OPENAI_API_KEY"])

def start_ui():
    st.set_page_config(page_title="PDF Chatbot")
    st.header("RAG-based LLM for PDF Analysis", divider='rainbow')

    if "messages" not in st.session_state:
        init_messages = [
            {
                "role": "ai",
                "content": "Hello! Good to See You. Malini this side, May I Know how to help you!"
            }
        ]
        st.session_state.messages = init_messages

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if promt := st.chat_input("Ask me anything from the Document."):
        with st.chat_message("human"):
            st.markdown(promt)
        st.session_state.messages.append({"role" : "human", "content" : promt})
        asyncio.run(process_input(promt))


    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files Here", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = read_pdf(pdf_docs)
                text_chunks = split_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

def main():
    start_ui()

if __name__ == "__main__":
    main()