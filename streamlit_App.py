import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os
import tempfile
import torch

st.set_page_config(page_title="CHATBOT WITH RAG", layout="wide")

st.markdown("""
## Document : Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, using zephyr 7b beta LLM Model. It processes uploaded PDF documents by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:


1. *Upload Your Documents*: The system accepts multiple PDF files at once, analyzing the content to provide comprehensive insights.

2. *Ask a Question*: After processing the documents, ask any question related to the content of your uploaded documents for a precise answer.
""")

@st.cache_resource
def load_llm_model():
    local_llm = r"C:\rag-chatbot"
    config = {
        'max_new_tokens': 1024,
        'repetition_penalty': 1.1,
        'temperature': 0.1,
        'top_k': 100,
        'top_p': 0.95,
        'stream': True,
        ##'gpu_layers': 50,  # Use 50 layers on GPU
        'threads': os.cpu_count()  # Use all available CPU threads
    }

    llm = CTransformers(
        model=local_llm,
        model_type="mistral",
        **config
    )

    print("LLM Initialized...")
    return llm

@st.cache_resource
def load_embeddings():
    local_model_dir =r"C:rag-chatbot\local_model"
    ##device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=local_model_dir,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
    return embeddings

def get_pdf_text(pdf_docs):
    all_documents = []
    for pdf_file in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getbuffer())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)
        os.remove(tmp_file_path)  # Clean up temporary file
    return all_documents

def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    for doc in docs:
        chunks.extend(text_splitter.split_text(doc.page_content))
    return chunks

def get_vector_store(text_chunks, embeddings):
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    vector_store = Chroma.from_documents(documents, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="stores/pet_cosine")
    return vector_store

def get_response(user_question, retriever, llm):
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain_type_kwargs = {"prompt": prompt}

    query = user_question
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    return response

def user_input(user_question, llm):
    embeddings = load_embeddings()
    load_vector_store = Chroma(persist_directory="stores/pet_cosine", embedding_function=embeddings)
    retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})

    response = get_response(user_question, retriever, llm)
    st.write("Reply: ", response)

def main():
    st.header("AI RAG chatbot💁")
    llm = load_llm_model()

    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    if user_question:  # Ensure API key and user question are provided
        user_input(user_question, llm)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                docs = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(docs)
                embeddings = load_embeddings()
                get_vector_store(text_chunks, embeddings)
                st.success("Done")

if __name__ == "__main__":
    main()