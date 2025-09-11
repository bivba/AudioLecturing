import os
import pickle
import threading
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain.schema import Document

class VectorDB:
    def __init__(self, index_path='faiss_index_cohere', embedding_model='embed-multilingual-v3.0'):
        self.index_path = index_path
        self.embedding_model = CohereEmbeddings(model=embedding_model, cohere_api_key=os.getenv("COHERE_API_KEY"))
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n## ", "\n### ", "\n#### ", "\n", " ", ""]
        )
        self.vectorstore = None
        self.load_or_create_index()

    def load_or_create_index(self):
        if os.path.exists(self.index_path):
            try:
                self.vectorstore = FAISS.load_local(self.index_path, self.embedding_model, allow_dangerous_deserialization=True)
                print("Loaded existing FAISS index.")
            except Exception as e:
                print(f"Error loading index: {e}. Creating new one.")
                self.vectorstore = None
        else:
            self.vectorstore = None
            print("No existing index found. Will create on first ingestion.")

    def add_document(self, text, metadata=None):
        if metadata is None:
            metadata = {}
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
        else:
            self.vectorstore.add_documents(documents)

        self.save_index()
        print(f"Added {len(chunks)} chunks to vector store.")

    def search(self, query, k=5):
        if self.vectorstore is None:
            return []
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def save_index(self):
        if self.vectorstore:
            self.vectorstore.save_local(self.index_path)
            print("Index saved.")
    
    def info(self):
        """Debug info about loaded embedding/index status."""
        print("---- VectorDB info ----")
        print("index_path:", self.index_path)
        print("embedding model:", type(self.embedding_model), getattr(self.embedding_model, "model_name", None))
        print("vectorstore is None?:", self.vectorstore is None)
        if self.vectorstore is not None:
            idx = getattr(self.vectorstore, "index", None)
            if idx is not None:
                ntotal = getattr(idx, "ntotal", None)
                print("FAISS index ntotal:", ntotal)
            docstore = getattr(self.vectorstore, "docstore", None)
            if docstore is not None:
                try:
                    n_docs = len(getattr(docstore, "_dict", getattr(docstore, "dict", {})))
                except Exception:
                    n_docs = "unknown"
                print("docstore size:", n_docs)
        print("------------------------")

_instance = None
_instance_lock = threading.Lock()

def get_vector_db():
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = VectorDB()
    return _instance