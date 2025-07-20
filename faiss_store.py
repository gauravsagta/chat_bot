# faiss_store.py

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from embedding_generator import EmbeddingGenerator

load_dotenv()

class FAISSStore:
    def __init__(self):
        self.index_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
        self.model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)

    def store(self):
        print("[INFO] Generating text chunks and embeddings using EmbeddingGenerator...")
        generator = EmbeddingGenerator()
        chunks = generator.processor.load_and_split()  # reuse the chunks from processor
        print(f"[INFO] Got {len(chunks)} text chunks.")

        print("[INFO] Creating FAISS index from chunks and embeddings...")
        db = FAISS.from_documents(chunks, self.embedding_model)

        db.save_local(self.index_path)
        print(f"[INFO] FAISS index saved at: {self.index_path}")


# Run store logic
vectorstore = FAISSStore()
vectorstore.store()
