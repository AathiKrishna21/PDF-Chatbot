# PDF Chatbot

A Streamlit-based chatbot that leverages Retrieval-Augmented Generation (RAG) and Large Language Models (LLMs) to answer questions from uploaded PDF documents. This project uses LangChain for document processing and ChromaDB for vector storage.

---

## Features

- **PDF Upload:** Upload a PDF and interact with its content via chat.
- **RAG-based LLM:** Uses Retrieval-Augmented Generation for accurate, context-aware answers.
- **Chat History:** Maintains conversation context for better responses.
- **Embeddings & Vector Store:** Uses local embeddings and ChromaDB for efficient document retrieval.

---

## Demo

![PDF Chatbot Demo]
*Upload a PDF, ask questions, and get instant answers!*

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/pdf-chatbot.git
    cd pdf-chatbot
    ```

2. **Create a virtual environment (recommended):**
    ```bash
    python -m venv chatbot
    chatbot\Scripts\activate  # On Windows
    # source chatbot/bin/activate  # On Linux/Mac
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Environment Variables

Some features (such as using OpenAI or other API-based LLMs) may require API keys or other secrets.  
These should be stored in a `.env` file in your project root.

**Example `.env` file:**
```
# mode as remote/local
MODEL_MODE=local

# MODEL_MODE=local
LLM_MODEL_NAME=meta-llama-3.1-8b-instruct
LLM_BASE_URL=http://10.2.0.2:3333/v1

# IF MODEL_MODE=remote
OPENAI_API_KEY=your_openai_api_key_here
# Add other environment variables as needed
```

**Instructions:**
1. Create a file named `.env` in the project root directory.
2. Add your API keys and other secrets as key-value pairs (see example above).
3. The application will automatically load these variables if you use [python-dotenv](https://pypi.org/project/python-dotenv/) or similar packages in your code.


## Usage

1. **Run the Streamlit app:**
    ```bash
    streamlit run main.py
    ```

2. **Open your browser:**  
   Visit [http://localhost:8501](http://localhost:8501) to use the chatbot.

3. **Upload a PDF:**  
   Use the sidebar to upload your PDF file.

4. **Chat:**  
   Ask questions about the uploaded document in the chat input box.

---

## Project Structure

```
.
├── main.py                # Streamlit app entry point
├── embeddings.py          # Embedding creation logic
├── vector_store.py        # ChromaDB setup and management
├── chain.py               # LLM chain construction
├── requirements.txt       # Python dependencies
```

---

## Requirements

See [`requirements.txt`](requirements.txt):

- streamlit
- langchain
- chromadb
- langchain-community
- langchain-core
- langchain-openai

---

## Customization

- **Embeddings:**  
  Modify `embeddings.py` to change embedding models or providers.

- **LLM Chain:**  
  Update `chain.py` to use different LLMs or prompt templates.

- **Vector Store:**  
  Tweak `vector_store.py` for different vector DBs or settings.

---

## Troubleshooting

- **Missing Keys Error:**  
  Ensure the input dictionary for the chain includes all required keys (`question`, `chat_history`).

- **PDF Not Loading:**  
  Make sure the uploaded file is a valid PDF and dependencies are installed.

- **ChromaDB Issues:**  
  Check that `chromadb` is installed and compatible with your Python version.

---

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)

---

## Author

- [Your Name](https://github.com/Aathikrishna21)