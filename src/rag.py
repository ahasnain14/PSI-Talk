# src/rag.py
import os
import json
from typing import List, Optional, Dict, Any

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from src.rag_loader import build_rag_docs
from src.embedder import (
   vec_store,
   db_dir as EMBEDDINGS_DIR,
)
from qgenie.integrations.langchain import QGenieChat

# Retrieval defaults
DEFAULT_K = 8                  # increased from 3 for better recall
DEFAULT_THRESHOLD = 0.05       # lowered from 0.1 so more docs are surfaced

# LLM defaults
MODEL = "anthropic::claude-4-5-sonnet"
MAX_TOKENS = 2048
TEMPERATURE = 0.3
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.1

# Paths
JSON_PATH = os.path.join("data", "psi_events.json")


def create_documents(json_path: str) -> List[Document]:
   """Build RAG docs from JSON using build_rag_docs."""
   if not os.path.exists(json_path):
       raise FileNotFoundError(f"JSON file not found at {json_path}")
   docs = build_rag_docs(JSON_PATH=json_path)
   if not docs:
       raise ValueError(f"No documents were created from {json_path}")
   return docs


def query_vector_store(
   store_name: str,
   query: str,
   embedding_function,
   k: int = DEFAULT_K,
   threshold: float = DEFAULT_THRESHOLD,
   base_dir: str = EMBEDDINGS_DIR,
) -> List[Document]:
   """
   Query the persisted vector store using similarity score threshold.

   Falls back to a broader fetch (threshold=0.0) if nothing is returned
   above the given threshold, so intricate/specific queries always get
   candidate documents.
   """
   persistent_directory = os.path.join(base_dir, store_name)

   if not os.path.exists(persistent_directory):
       raise FileNotFoundError(
           f"Vector store {store_name} does not exist at {persistent_directory}"
       )

   db = Chroma(
       persist_directory=persistent_directory,
       embedding_function=embedding_function,
   )

   # Primary retrieval
   retriever = db.as_retriever(
       search_type="similarity_score_threshold",
       search_kwargs={"k": k, "score_threshold": threshold},
   )
   relevant_docs = retriever.invoke(query)

   # Fallback: if nothing matched, pull top-k with no threshold so the LLM
   # still has raw context to work with rather than an empty list
   if not relevant_docs:
       fallback_retriever = db.as_retriever(
           search_type="similarity_score_threshold",
           search_kwargs={"k": k, "score_threshold": 0.0},
       )
       relevant_docs = fallback_retriever.invoke(query)

   return relevant_docs


def _load_raw_events(json_path: str) -> List[Dict[str, Any]]:
   """Load raw events from the JSON file for direct look-up."""
   if not json_path or not os.path.exists(json_path):
       return []
   try:
       with open(json_path, "r") as f:
           data = json.load(f)
       return data.get("events", [])
   except Exception:
       return []


def _keyword_fallback_docs(
   query: str,
   json_path: str,
   max_events: int = 10,
) -> List[Document]:
   """
   Very cheap keyword search over raw events as an additional fallback.

   Looks for query tokens (lowercased) inside the JSON-serialised event.
   Returns matched events as Documents so they can be injected into the
   LLM prompt alongside the vector results.
   """
   from src.rag_loader import event_to_text

   events = _load_raw_events(json_path)
   if not events:
       return []

   tokens = [t.lower() for t in query.split() if len(t) > 2]
   if not tokens:
       return []

   matched = []
   for ev in events:
       blob = json.dumps(ev).lower()
       if any(tok in blob for tok in tokens):
           matched.append(ev)

   matched = matched[:max_events]
   docs = []
   for ev in matched:
       try:
           text = event_to_text(ev)
           docs.append(
               Document(
                   page_content=text,
                   metadata={
                       "kind": ev.get("kind"),
                       "timestamp": ev.get("timestamp_sec"),
                       "source": "keyword_fallback",
                   },
               )
           )
       except Exception:
           continue
   return docs


def build_combined_input(
   query: str,
   relevant_docs: List[Document],
   all_events_summary: Optional[str] = None,
) -> str:
   """
   Combine query + relevant docs into a single LLM input.

   The prompt is designed to push the model to reason over the actual
   timestamps and numbers rather than giving up with "I'm not sure".
   """
   if relevant_docs:
       docs_text = "\n\n".join(
           [
               f"[Event {i+1} | kind={doc.metadata.get('kind','?')} "
               f"ts={doc.metadata.get('timestamp','?')}]\n{doc.page_content}"
               for i, doc in enumerate(relevant_docs)
           ]
       )
   else:
       docs_text = "(No directly matching documents were retrieved.)"

   header = (
       "You are an expert Linux performance engineer analysing PSI "
       "(Pressure Stall Information) and OOM event logs.\n\n"
       "INSTRUCTIONS:\n"
       "- Answer the question using ONLY the event documents provided below.\n"
       "- Each document contains a timestamp (ts=) and event kind. "
       "Use these timestamps to answer time-specific questions precisely.\n"
       "- If a specific process, timestamp, or value is mentioned in the "
       "documents, extract and quote it directly.\n"
       "- If the exact answer is not in the documents, say what you CAN "
       "infer from the closest matching events, and state clearly what is "
       "uncertain. Do NOT simply say 'I'm not sure' without attempting "
       "to reason over the data.\n"
       "- Format numbers with units (seconds, kB, ms, %).\n\n"
   )

   combined = (
       f"{header}"
       f"QUESTION: {query}\n\n"
       f"RELEVANT EVENT DOCUMENTS ({len(relevant_docs)} retrieved):\n"
       f"{docs_text}\n"
   )

   if all_events_summary:
       combined += f"\nADDITIONAL CONTEXT:\n{all_events_summary}\n"

   return combined


def get_llm_response(
   prompt: str,
   api_key: str,
   model: str = MODEL,
   max_tokens: int = MAX_TOKENS,
   temperature: float = TEMPERATURE,
) -> str:
   """
   Get response from LLM.

   Args:
       prompt: Input prompt
       api_key: QGenie API key
       model: Model identifier
       max_tokens: Maximum tokens to generate
       temperature: Sampling temperature

   Returns:
       LLM response content

   Raises:
       ValueError: If API key is not available
   """
   if not api_key:
       raise ValueError("api_key must be provided to get_llm_response")

   model_instance = QGenieChat(api_key=api_key, model=model)
   messages = [HumanMessage(content=prompt)]

   result = model_instance.invoke(
       messages,
       max_tokens=max_tokens,
       repetition_penalty=REPETITION_PENALTY,
       temperature=temperature,
       top_k=TOP_K,
       top_p=TOP_P,
   )

   return result.content


def run_query(
   query: str,
   api_key: str,
   json_path: str = JSON_PATH,
   store_name: str = "psi_vector_db",
   embedding_function=None,
   k: int = DEFAULT_K,
   threshold: float = DEFAULT_THRESHOLD,
   temperature: float = TEMPERATURE,
   verbose: bool = True,
) -> str:
   """
   Full RAG pipeline: retrieve → keyword fallback → LLM answer.

   Args:
       query: User's question
       api_key: QGenie API key
       json_path: Path to parsed events JSON
       store_name: Vector store name
       embedding_function: Embedding function to use
       k: Number of documents to retrieve
       threshold: Similarity threshold
       temperature: LLM sampling temperature
       verbose: Whether to print progress

   Returns:
       LLM response string
   """
   if not api_key:
       raise ValueError("api_key is required")

   # 1) Build docs & ensure vector store
   docs = create_documents(json_path)
   if embedding_function is None:
       from src.embedder import get_embeddings_fn
       embedding_function = get_embeddings_fn(api_key)

   vec_store(docs, embedding_function, store_name)

   # 2) Vector retrieval
   relevant_docs = query_vector_store(
       store_name=store_name,
       query=query,
       embedding_function=embedding_function,
       k=k,
       threshold=threshold,
       base_dir=EMBEDDINGS_DIR,
   )

   # 3) Keyword fallback – merge unique docs
   kw_docs = _keyword_fallback_docs(query, json_path, max_events=k)
   existing_contents = {d.page_content for d in relevant_docs}
   for kd in kw_docs:
       if kd.page_content not in existing_contents:
           relevant_docs.append(kd)
           existing_contents.add(kd.page_content)

   if verbose:
       print(f"Retrieved {len(relevant_docs)} documents for query: '{query}'")

   # 4) Build prompt & call LLM
   combined_input = build_combined_input(query, relevant_docs)
   response = get_llm_response(
       combined_input, api_key=api_key, temperature=temperature
   )

   if verbose:
       print(response)

   return response