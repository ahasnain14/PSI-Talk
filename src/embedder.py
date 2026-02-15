# src/embedder.py
import os
import shutil
from typing import List
from langchain_chroma import Chroma
from qgenie.integrations.langchain import QGenieEmbeddings
from langchain_core.documents import Document

# Configuration
store_name = "psi_vector_db"
db_dir = "embeddings"


def get_embeddings_fn(api_key: str) -> QGenieEmbeddings:
   """
   Create an embeddings function with the given API key.

   Args:
       api_key: QGenie API key

   Returns:
       QGenieEmbeddings instance
   """
   if not api_key:
       raise ValueError("api_key must be provided to get_embeddings_fn")
   return QGenieEmbeddings(api_key=api_key)


def _normalize_docs(docs: List) -> List[Document]:
   """
   Ensure every item is a langchain Document.

   Args:
       docs: List of potential Document objects or tuples

   Returns:
       List of proper Document objects
   """
   normalized = []
   for d in docs:
       if isinstance(d, tuple) and len(d) == 1:
           d = d[0]
       if isinstance(d, list) and len(d) == 1:
           d = d[0]
       if not isinstance(d, Document):
           d = Document(page_content=str(d), metadata={})
       normalized.append(d)
   return normalized


def _sanitize_docs(docs: List[Document]) -> List[Document]:
   """
   Sanitize document metadata to avoid Chroma storage issues.

   Args:
       docs: List of Document objects

   Returns:
       List of Documents with sanitized metadata
   """
   safe_docs = []
   for d in docs:
       safe_metadata = {
           "kind": d.metadata.get("kind"),
           "timestamp": d.metadata.get("timestamp"),
       }
       safe_docs.append(
           Document(page_content=d.page_content, metadata=safe_metadata)
       )
   return safe_docs


def vec_store(
   docs: List[Document],
   embeddings,
   store_name: str,
   base_dir: str = db_dir,
   force_recreate: bool = False,
) -> Chroma:
   """
   Create or load a vector store from documents.

   Args:
       docs: List of Document objects to embed
       embeddings: Embedding function
       store_name: Name of the vector store
       base_dir: Base directory for storing embeddings
       force_recreate: If True, recreate the store even if it exists

   Returns:
       Chroma vector store instance
   """
   persistent_directory = os.path.join(base_dir, store_name)
   os.makedirs(base_dir, exist_ok=True)

   if not os.path.exists(persistent_directory) or force_recreate:
       if force_recreate and os.path.exists(persistent_directory):
           shutil.rmtree(persistent_directory)

       docs_norm = _normalize_docs(docs)
       docs_simple = _sanitize_docs(docs_norm)

       if not docs_simple:
           raise ValueError("No valid documents to embed")

       db = Chroma.from_documents(
           docs_simple, embeddings, persist_directory=persistent_directory
       )
       return db
   else:
       return Chroma(
           persist_directory=persistent_directory,
           embedding_function=embeddings,
       )


def rename_vector_store(old_name: str, new_name: str, base_dir: str = db_dir) -> bool:
   """
   Rename a vector store directory.

   Args:
       old_name: Current store name
       new_name: New store name
       base_dir: Base directory for embeddings

   Returns:
       True if successful, False otherwise
   """
   old_path = os.path.join(base_dir, old_name)
   new_path = os.path.join(base_dir, new_name)

   if not os.path.exists(old_path):
       return False
   if os.path.exists(new_path):
       return False

   os.rename(old_path, new_path)
   return True


def delete_vector_store(store_name: str, base_dir: str = db_dir) -> bool:
   """
   Delete a vector store directory.

   Args:
       store_name: Name of the store to delete
       base_dir: Base directory for embeddings

   Returns:
       True if successful, False otherwise
   """
   path = os.path.join(base_dir, store_name)
   if not os.path.exists(path):
       return False
   shutil.rmtree(path)
   return True


def load_vector_store(
   store_name: str, embeddings, base_dir: str = db_dir
) -> Chroma:
   """
   Load an existing vector store.

   Args:
       store_name: Name of the vector store
       embeddings: Embedding function
       base_dir: Base directory for embeddings

   Returns:
       Chroma vector store instance

   Raises:
       FileNotFoundError: If vector store doesn't exist
   """
   persistent_directory = os.path.join(base_dir, store_name)
   if not os.path.exists(persistent_directory):
       raise FileNotFoundError(
           f"Vector store '{store_name}' not found at {persistent_directory}"
       )
   return Chroma(
       persist_directory=persistent_directory, embedding_function=embeddings
   )


def list_vector_stores(base_dir: str = db_dir) -> List[str]:
   """
   List all available vector stores.

   Args:
       base_dir: Base directory for embeddings

   Returns:
       List of vector store names
   """
   if not os.path.exists(base_dir):
       return []

   stores = []
   for item in os.listdir(base_dir):
       item_path = os.path.join(base_dir, item)
       if os.path.isdir(item_path) and os.path.exists(
           os.path.join(item_path, "chroma.sqlite3")
       ):
           stores.append(item)
   return stores