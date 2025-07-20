# embedding_generator.py

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from document_processor import DocumentProcessor

load_dotenv()

class EmbeddingGenerator:
    def __init__(self):
        model_name = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_name)
        self.processor = DocumentProcessor()

    def run(self):
        chunks = self.processor.load_and_split()
        print(f"[INFO] Number of chunks: {len(chunks)}")
        texts = [doc.page_content for doc in chunks]
        embeddings = self.model.encode(texts)
        print(f"[INFO] Generated {len(embeddings)} embeddings.")
        return embeddings
