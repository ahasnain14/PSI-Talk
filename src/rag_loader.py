# src/rag_loader.py
import json
import os
from typing import List, Dict, Any
from langchain_core.documents import Document 
from langchain_community.document_loaders import JSONLoader

def load_events(json_path: str):
  """
  Load structured PSI events using JSONLoader
  
  Args:
      json_path: Path to the JSON file containing parsed events
  
  Returns:
      List of Document objects
  
  Raises:
      FileNotFoundError: If JSON file doesn't exist
      ValueError: If JSON structure is invalid
  """
  if not os.path.exists(json_path):
      raise FileNotFoundError(f"JSON file not found: {json_path}")
  
  try:
      loader = JSONLoader(
          file_path=json_path,
          jq_schema=".events[]",
          text_content=False
      )
      return loader.load()
  except Exception as e:
      raise ValueError(f"Error loading events from {json_path}: {str(e)}")

def summarize_tasks(tasks: List[Dict], top_n: int = 10) -> str:
  """
  Summarize top resource-consuming tasks.
  
  Args:
      tasks: List of task dictionaries
      top_n: Number of top tasks to include
  
  Returns:
      Human-readable summary string
  """
  if not tasks:
      return "No dominant processes were recorded. "
  
  tasks = sorted(tasks, key=lambda t: t.get("score", 0), reverse=True)[:top_n]

  parts = []
  for t in tasks:
      comm = t.get("comm", "?")
      pid = t.get("pid", "?")
      rss_kb = t.get("rss_kb", "?")
      cpu_ms = t.get("cpu_ms", "?")
      parts.append(f"{comm} (pid {pid}) used {rss_kb} kB RAM and {cpu_ms} ms CPU")
  
  return "Top contributing processes were: " + "; ".join(parts) + "."

def pressure_event_text(ev: Dict[str, Any]) -> str:
  """Convert pressure event to human-readable text"""
  psi = ev.get("psi", {})
  ts = float(ev.get("timestamp_sec", 0.0) or 0.0)
  cpu = psi.get("cpu", "?")
  mem = psi.get("mem", "?")
  io = psi.get("io", "?")
  
  base = (
      f"At {ts:.6f} seconds the system experienced high resource pressure. "
      f"CPU pressure was {cpu} percent, "
      f"memory pressure was {mem} percent, "
      f"and IO pressure was {io} percent. "
  )
  
  return base + summarize_tasks(ev.get("tasks", []))

def oom_kill_text(ev: Dict[str, Any]) -> str:
  """Convert OOM kill event to human-readable text"""
  ts = float(ev.get('timestamp_sec', 0.0) or 0.0)
  task = ev.get('task', '?')
  pid = ev.get('pid', '?')
  constraint = ev.get('constraint', ev.get('details', {}).get('constraint', '?'))
  
  return (
      f"At {ts:.6f} seconds the kernel initiated an out of memory kill. "
      f"The process {task} with pid {pid} was selected. "
      f"Constraint: {constraint}."
  )

def oom_killed_process_text(ev: Dict[str, Any]) -> str:
  """Convert OOM killed process event to human-readable text"""
  ts = float(ev.get('timestamp_sec', 0.0) or 0.0)
  comm = ev.get('comm', '?')
  pid = ev.get('pid', '?')
  summary = ev.get('summary', '')
  
  return (
      f"At {ts:.6f} seconds the process {comm} "
      f"pid {pid} was killed due to memory exhaustion. "
      f"Details: {summary}."
  )

def oom_reaper_text(ev: Dict[str, Any]) -> str:
  """Convert OOM reaper event to human-readable text"""
  ts = float(ev.get('timestamp_sec', 0.0) or 0.0)
  comm = ev.get('comm', '?')
  pid = ev.get('pid', '?')
  
  return (
      f"At {ts:.6f} seconds the OOM reaper reclaimed memory "
      f"from process {comm} with pid {pid}."
  )

def event_to_text(ev: Dict[str, Any]) -> str:
  """
  Convert any event type to human-readable text.
  
  Args:
      ev: Event dictionary
  
  Returns:
      Human-readable event description
  """
  kind = ev.get("kind")

  if kind == "pressure_high":
      return pressure_event_text(ev)
  if kind == "oom_kill":
      return oom_kill_text(ev)
  if kind == "oom_killed_process":
      return oom_killed_process_text(ev)
  if kind == "oom_reaper":
      return oom_reaper_text(ev)

  # Fallback to JSON for unknown event types
  return json.dumps(ev, indent=2)

def _extract_event_from_doc(doc: Document) -> Dict[str, Any]:
  """
  Extract event object from a JSONLoader Document.
  
  JSONLoader(text_content=False) can place the object in:
    - doc.metadata["json"] (most common)
    - doc.metadata (direct placement)
    - doc.page_content (as JSON string)
  
  Args:
      doc: LangChain Document object
  
  Returns:
      Event dictionary
  
  Raises:
      ValueError: If event cannot be extracted
  """
  md = doc.metadata or {}

  # Most common path: metadata["json"]
  if isinstance(md, dict) and isinstance(md.get("json"), dict):
      return md["json"]

  # Sometimes metadata itself is the event
  if isinstance(md, dict) and md.get("kind"):
      return md

  # Last resort: parse page_content as JSON
  if doc.page_content:
      try:
          parsed = json.loads(doc.page_content)
          if isinstance(parsed, dict) and parsed.get("kind"):
              return parsed
      except json.JSONDecodeError:
          pass

  raise ValueError(
      f"Could not extract event object from Document. "
      f"Metadata keys: {list(md.keys()) if isinstance(md, dict) else 'not a dict'}"
  )

def build_rag_docs(JSON_PATH: str) -> List[Document]:
  """
  Build RAG-ready documents from parsed PSI events.
  
  Args:
      JSON_PATH: Path to parsed events JSON file
  
  Returns:
      List of Document objects with human-readable content and metadata
  """
  raw_docs = load_events(JSON_PATH)
  docs: List[Document] = []

  for doc in raw_docs:
      try:
          ev = _extract_event_from_doc(doc)
          text = event_to_text(ev)
      
          # Extract top 5 tasks for metadata
          top_tasks = [
              {
                  "pid": t.get("pid"),
                  "comm": t.get("comm"),
                  "score": t.get("score"),
                  "rss_kb": t.get("rss_kb"),
                  "io_kb": t.get("io_kb"),
                  "cpu_ms": t.get("cpu_ms"),
              }
              for t in ev.get("tasks", [])[:5]
          ]
          
          docs.append(
              Document(
                  page_content=text,
                  metadata={
                      "kind": ev.get("kind"),
                      "timestamp": ev.get("timestamp_sec"),
                      "top_tasks": top_tasks,
                  },
              )
          )
      except Exception as e:
          print(f"Warning: Skipping document due to error: {str(e)}")
          continue

  return docs  

if __name__ == "__main__":
  # Test with default path
  JSON_PATH = os.path.join("data", "psi_events.json")
  
  if not os.path.exists(JSON_PATH):
      print(f"Error: {JSON_PATH} not found. Please run parser.py first.")
  else:
      docs = build_rag_docs(JSON_PATH=JSON_PATH)
      print(f"Built {len(docs)} RAG docs")
      if docs:
          print("\nSample document:")
          print(f"Content: {docs[0].page_content}")
          print(f"Metadata: {docs[0].metadata}")