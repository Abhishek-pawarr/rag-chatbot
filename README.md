##    chatbot with RAG

## Overview

ROBO is an interactive chatbot application that uses Retrieval-Augmented Generation (RAG) to provide precise answers to questions about the content of uploaded PDF documents. It leverages advanced machine learning models, including Google's Generative AI model Gemini-PRO, to deliver contextually relevant responses efficiently.

## Features

- **Multiple PDF Uploads**: Accepts multiple PDF files for comprehensive content analysis.
- **Intelligent Question Answering**: Provides accurate answers to user queries based on the content of the uploaded documents.
- **Efficient Document Processing**: Breaks down documents into manageable chunks and creates a searchable vector store for quick retrieval of relevant information.
- **User-Friendly Interface**: Simple and intuitive interface built with Streamlit for easy interaction.

## How It Works

1. **Upload Your Documents**: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.
2. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.

## Installation

1. Clone the repository:
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure you have the required local models downloaded and accessible at `C:\rag-chatbot` and `C:\rag-chatbot\local_model`.

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501` to access the application.

## Code Structure

- **`streamlit_App.py`**: The main application file containing the Streamlit interface and logic for document processing and question answering.
- **Functions**:
  - `load_llm_model()`: Loads the local LLM model with the specified configuration.
  - `load_embeddings()`: Loads the embeddings model.
  - `get_pdf_text(pdf_docs)`: Extracts text from the uploaded PDF documents.
  - `get_text_chunks(docs)`: Splits the extracted text into manageable chunks.
  - `get_vector_store(text_chunks, embeddings)`: Creates a vector store from the text chunks.
  - `get_response(user_question, retriever, llm)`: Generates a response to the user's question based on the retrieved documents.
  - `user_input(user_question, llm)`: Handles user input and generates responses.
  - `main()`: The main function to run the Streamlit application.

## Notes

- Ensure the local models are correctly downloaded and placed in the specified directories.
- Adjust the paths and configurations as needed for your environment.
- The application uses GPU for model inference if available. Ensure the necessary GPU drivers and libraries are installed.


## Acknowledgements

- [Streamlit](https://streamlit.io/) for providing the web application framework.
- [LangChain](https://www.langchain.com/) for the document processing and vector store utilities.
- [Hugging Face](https://huggingface.co/) for the embedding models.

