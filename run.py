import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from embedding_generator import EmbeddingGenerator
from langchain_community.llms import Ollama


# Load environment variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index")
llm = Ollama(model="llama3.2")
# Load models
@st.cache_resource
def load_models():
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    return vector_store

@st.cache_resource
def load_query_embedder():
    return EmbeddingGenerator().model  # using the SentenceTransformer instance from your class

# Streamlit UI
st.title("📚 FAISS RAG Chatbot")
st.markdown("Ask a question based on your indexed documents.")

query = st.text_input("Enter your query:")
print(query)

if query:
    with st.spinner("Embedding and searching..."):
        vector_store = load_models()
        retriever = vector_store.as_retriever(search_kwargs={"k":5,"score_threshold":0.9})
        documents=retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt="""You are an AI assistance designed to answer the user query based on the retrieved context and the user query:
        User query: {query}
        Retrieved context: {context}
         """
        formatted_prompt = prompt.format(query=query, context=context)
        print(formatted_prompt)
        result=llm.invoke(formatted_prompt)
        st.write(result)
