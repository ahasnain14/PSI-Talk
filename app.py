import os
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

import io
import re

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()
_API_KEY = os.getenv("API_KEY", "")

from src.parser import parse_log
from src.rag_loader import build_rag_docs
from src.embedder import (
   get_embeddings_fn,
   vec_store,
   rename_vector_store,
   delete_vector_store,
   db_dir as EMBEDDINGS_DIR,
   list_vector_stores,
)
from src.rag import (
   query_vector_store,
   build_combined_input,
   get_llm_response,
   _keyword_fallback_docs,
   DEFAULT_K,
   DEFAULT_THRESHOLD,
   MODEL,
   TEMPERATURE,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
   page_title="PSI-Talk",
   page_icon="📊",
   layout="wide",
   initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---- global resets ---- */
    .block-container { padding-top: 1.5rem; }

    /* ---- report box ---- */
    .report-box {
        background-color: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        color: inherit;
        backdrop-filter: blur(2px);
    }

    /* ---- store list items ---- */
    .store-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.45rem 0.6rem;
        border-radius: 0.5rem;
        margin-bottom: 0.25rem;
        cursor: pointer;
        transition: background 0.15s;
    }

    /* --- Sidebar store row: force vertical centering all the way down --- */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
        align-items: center !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"] {
        display: flex !important;
        align-items: center !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stColumn"] > div {
        display: flex !important;
        align-items: center !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button {
        height: 2.2rem !important;
        min-height: 2.2rem !important;
        padding: 0 !important;
        margin: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 1 !important;
        font-size: 0.95rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
        line-height: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Metadata helpers ──────────────────────────────────────────────────────────
STORE_META_FILE = os.path.join(EMBEDDINGS_DIR, ".store_metadata.json")

def _load_store_meta() -> Dict[str, Any]:
   if os.path.exists(STORE_META_FILE):
       try:
           with open(STORE_META_FILE) as f:
               return json.load(f)
       except Exception:
           pass
   return {}

def _save_store_meta(meta: Dict[str, Any]):
   os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
   with open(STORE_META_FILE, "w") as f:
       json.dump(meta, f, indent=2)


def _load_events_from_json(json_path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        data["events"].sort(key=lambda e: e.get("timestamp_sec", 0.0))
        return data
    except Exception:
        return None
    
def _get_display_name(store_id: str) -> str:
   meta = _load_store_meta()
   return meta.get(store_id, {}).get("display_name", store_id)

def _set_display_name(store_id: str, display_name: str):
   meta = _load_store_meta()
   if store_id not in meta:
       meta[store_id] = {}
   meta[store_id]["display_name"] = display_name
   _save_store_meta(meta)

def _register_store(store_id: str, display_name: str, log_filename: str):
   meta = _load_store_meta()
   meta[store_id] = {
       "display_name": display_name,
       "log_filename": log_filename,
       "created_at": datetime.now().isoformat(),
   }
   _save_store_meta(meta)

def _unregister_store(store_id: str):
   meta = _load_store_meta()
   meta.pop(store_id, None)
   _save_store_meta(meta)

def strip_markdown(text: str) -> str:
    """
    Convert common Markdown to plain text for PDF.
    (removes **bold**, *italic*, `code`, headings, links)
    """
    if not text:
        return ""

    t = text

    # Remove code fences
    t = re.sub(r"```.*?```", "", t, flags=re.DOTALL)

    # Inline code
    t = re.sub(r"`([^`]+)`", r"\1", t)

    # Bold/italic markers
    t = t.replace("**", "").replace("*", "").replace("__", "").replace("_", "")

    # Headings like #, ##, ###
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.MULTILINE)

    # Markdown links [text](url) -> text (url)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", t)

    # Bullet normalization
    t = t.replace("•", "-")

    return t.strip()

def _rename_store_meta(old_id: str, new_id: str):
   meta = _load_store_meta()
   if old_id in meta:
       meta[new_id] = meta.pop(old_id)
       _save_store_meta(meta)

def compute_metrics_summary(events_data: Dict[str, Any]) -> Dict[str, Any]:
    events = events_data.get("events", []) if events_data else []
    pressure_events = [e for e in events if e.get("kind") == "pressure_high"]
    oom_events = [e for e in events if e.get("kind") in ["oom_kill", "oom_killed_process", "oom_reaper"]]

    def _avg(lst, key1, key2=None):
        vals = [e.get(key1, {}).get(key2, 0) if key2 else e.get(key1, 0) for e in lst]
        return (sum(vals) / len(vals)) if vals else 0.0

    avg_cpu = _avg(pressure_events, "psi", "cpu")
    avg_mem = _avg(pressure_events, "psi", "mem")
    avg_io  = _avg(pressure_events, "psi", "io")

    max_cpu = max((e.get("psi", {}).get("cpu", 0) for e in pressure_events), default=0)
    max_mem = max((e.get("psi", {}).get("mem", 0) for e in pressure_events), default=0)
    max_io  = max((e.get("psi", {}).get("io", 0) for e in pressure_events), default=0)

    return {
        "total_events": len(events),
        "pressure_events": len(pressure_events),
        "oom_events": len(oom_events),
        "avg_cpu": avg_cpu, "avg_mem": avg_mem, "avg_io": avg_io,
        "max_cpu": max_cpu, "max_mem": max_mem, "max_io": max_io,
    }

def pressure_timeline_png(events_data: Dict[str, Any]) -> Optional[bytes]:
    pressure_events = [e for e in events_data.get("events", []) if e.get("kind") == "pressure_high"]
    if not pressure_events:
        return None

    df = pd.DataFrame(
        {
            "time": [e.get("timestamp_sec", 0) for e in pressure_events],
            "cpu":  [e.get("psi", {}).get("cpu", 0) for e in pressure_events],
            "mem":  [e.get("psi", {}).get("mem", 0) for e in pressure_events],
            "io":   [e.get("psi", {}).get("io", 0) for e in pressure_events],
        }
    ).sort_values("time")

    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150)
    ax.plot(df["time"], df["cpu"], label="CPU")
    ax.plot(df["time"], df["mem"], label="Memory")
    ax.plot(df["time"], df["io"], label="I/O")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Pressure %")
    ax.set_title("System Pressure Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def generate_full_pdf_bytes(
    title: str,
    report_md: str,
    metrics: Dict[str, Any],
    chart_png: Optional[bytes],
) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Layout constants
    left = 0.75 * inch
    right = width - 0.75 * inch
    y = height - 0.75 * inch

    def new_page():
        nonlocal y
        c.showPage()
        y = height - 0.75 * inch

    def ensure_space(required):
        nonlocal y
        if y - required < 0.75 * inch:
            new_page()

    # ----- Title -----
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, title)
    y -= 0.35 * inch

    # ----- Metrics block -----
    ensure_space(1.2 * inch)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Key Metrics")
    y -= 0.2 * inch

    c.setFont("Helvetica", 10)
    lines = [
        f"Total Events: {metrics.get('total_events', 0)}",
        f"Pressure Events: {metrics.get('pressure_events', 0)}",
        f"OOM Events: {metrics.get('oom_events', 0)}",
        "",
        f"Avg CPU Pressure: {metrics.get('avg_cpu', 0):.1f}%   (Max: {metrics.get('max_cpu', 0)}%)",
        f"Avg Memory Pressure: {metrics.get('avg_mem', 0):.1f}% (Max: {metrics.get('max_mem', 0)}%)",
        f"Avg I/O Pressure: {metrics.get('avg_io', 0):.1f}%     (Max: {metrics.get('max_io', 0)}%)",
    ]
    for ln in lines:
        c.drawString(left, y, ln)
        y -= 0.18 * inch

    y -= 0.1 * inch

    # ----- Chart -----
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "Pressure Timeline")
    y -= 0.2 * inch

    if chart_png:
        ensure_space(3.0 * inch)
        img = ImageReader(io.BytesIO(chart_png))

        # Fit image within page width
        img_width = right - left
        img_height = 2.6 * inch
        c.drawImage(img, left, y - img_height, width=img_width, height=img_height, preserveAspectRatio=True, anchor='sw')
        y -= (img_height + 0.3 * inch)
    else:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(left, y, "No pressure data available to plot.")
        y -= 0.25 * inch

    # ----- AI Report -----
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, "AI Analysis Report")
    y -= 0.25 * inch

    report = strip_markdown(report_md)

    # Write wrapped text with page breaks
    c.setFont("Helvetica", 10)
    max_chars = 100  # conservative wrap for letter size

    for paragraph in (report or "").split("\n"):
        paragraph = paragraph.rstrip()
        if not paragraph:
            y -= 0.14 * inch
            continue

        words = paragraph.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 <= max_chars:
                line = f"{line} {w}".strip()
            else:
                ensure_space(0.25 * inch)
                c.drawString(left, y, line)
                y -= 0.18 * inch
                line = w

        if line:
            ensure_space(0.25 * inch)
            c.drawString(left, y, line)
            y -= 0.18 * inch

        y -= 0.08 * inch  # paragraph gap

    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
   defaults = {
       "api_key": _API_KEY,
       "embeddings_fn": None,
       "processed": False,
       "json_path": None,
       "store_name": None,
       "initial_report": None,
       "chat_history": [],
       "events_data": None,
       "temp_dir": None,
       "active_store": None,        # store currently selected in sidebar
       "rename_target": None,       # store being renamed
       "rename_value": "",
       "confirm_delete": None,      # store pending delete confirmation
   }
   for k, v in defaults.items():
       if k not in st.session_state:
           st.session_state[k] = v


# ── Chart ─────────────────────────────────────────────────────────────────────
def plot_pressure_over_time(events_data: Dict[str, Any]):
   pressure_events = [
       e for e in events_data["events"] if e.get("kind") == "pressure_high"
   ]
   if not pressure_events:
       st.info("No pressure data to plot")
       return

   df = pd.DataFrame(
       {
           "time": [e["timestamp_sec"] for e in pressure_events],
           "cpu": [e["psi"].get("cpu", 0) for e in pressure_events],
           "mem": [e["psi"].get("mem", 0) for e in pressure_events],
           "io": [e["psi"].get("io", 0) for e in pressure_events],
       }
   ).sort_values("time")

   fig, ax = plt.subplots(figsize=(10, 4))
   ax.plot(df["time"], df["cpu"], label="CPU")
   ax.plot(df["time"], df["mem"], label="Memory")
   ax.plot(df["time"], df["io"], label="I/O")
   ax.set_xlabel("Time (seconds)")
   ax.set_ylabel("Pressure %")
   ax.set_title("System Pressure Over Time")
   ax.legend()
   ax.grid(True, alpha=0.3)
   st.pyplot(fig)


# ── Metrics ───────────────────────────────────────────────────────────────────
def display_metrics(events_data: Dict[str, Any]):
   events = events_data.get("events", [])
   if not events:
       st.warning("No events to display")
       return

   pressure_events = [e for e in events if e.get("kind") == "pressure_high"]
   oom_events = [
       e
       for e in events
       if e.get("kind") in ["oom_kill", "oom_killed_process"]
   ]

   def _avg(lst, key1, key2=None):
       vals = [e.get(key1, {}).get(key2, 0) if key2 else e.get(key1, 0) for e in lst]
       return sum(vals) / len(vals) if vals else 0

   avg_cpu = _avg(pressure_events, "psi", "cpu")
   avg_mem = _avg(pressure_events, "psi", "mem")
   avg_io = _avg(pressure_events, "psi", "io")
   max_cpu = max((e.get("psi", {}).get("cpu", 0) for e in pressure_events), default=0)
   max_mem = max((e.get("psi", {}).get("mem", 0) for e in pressure_events), default=0)
   max_io = max((e.get("psi", {}).get("io", 0) for e in pressure_events), default=0)

   col1, col2, col3, col4 = st.columns(4)
   with col1:
       st.metric("Total Events", len(events))
       st.caption(f"Pressure: {len(pressure_events)} | OOM: {len(oom_events)}")
   with col2:
       st.metric("Avg CPU Pressure", f"{avg_cpu:.1f}%", delta=f"Max: {max_cpu}%" if max_cpu else None)
   with col3:
       st.metric("Avg Memory Pressure", f"{avg_mem:.1f}%", delta=f"Max: {max_mem}%" if max_mem else None)
   with col4:
       st.metric("Avg I/O Pressure", f"{avg_io:.1f}%", delta=f"Max: {max_io}%" if max_io else None)


# ── File processing ───────────────────────────────────────────────────────────
def process_log_file(uploaded_file):
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   session_dir = os.path.join("temp", timestamp)
   os.makedirs(session_dir, exist_ok=True)

   log_path = os.path.join(session_dir, uploaded_file.name)
   with open(log_path, "wb") as f:
       f.write(uploaded_file.getbuffer())

   with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
       lines = f.readlines()

   parsed_data = parse_log(lines)
   parsed_data["events"].sort(key=lambda e: e.get("timestamp_sec", 0.0))

   json_path = os.path.join(session_dir, "psi_events.json")
   with open(json_path, "w") as f:
       json.dump(parsed_data, f, indent=2)

   return session_dir, json_path, parsed_data


# ── Initial report ────────────────────────────────────────────────────────────
def generate_initial_report(events_data: Dict[str, Any], api_key: str) -> str:
   events = events_data.get("events", [])
   pressure_events = [e for e in events if e.get("kind") == "pressure_high"]
   oom_events = [
       e
       for e in events
       if e.get("kind") in ["oom_kill", "oom_killed_process", "oom_reaper"]
   ]

   if not events:
       return "No events found in the log file."

   timestamps = [e.get("timestamp_sec", 0) for e in events if e.get("timestamp_sec")]
   time_range = (
       f"{min(timestamps):.2f}s to {max(timestamps):.2f}s"
       if timestamps
       else "Unknown"
   )

   prompt = (
       f"Analyze the following PSI and OOM log summary and provide a concise executive report.\n\n"
       f"OVERVIEW:\n"
       f"- Total Events: {len(events)}\n"
       f"- Pressure Events: {len(pressure_events)}\n"
       f"- OOM Events: {len(oom_events)}\n"
       f"- Time Range: {time_range}\n\n"
   )

   if pressure_events:
       prompt += "KEY PRESSURE EVENTS:\n"
       for i, ev in enumerate(pressure_events[:5], 1):
           psi = ev.get("psi", {})
           ts = ev.get("timestamp_sec", 0)
           tasks = ev.get("tasks", [])[:3]
           prompt += (
               f"\n{i}. At {ts:.2f}s: CPU={psi.get('cpu')}%, "
               f"MEM={psi.get('mem')}%, IO={psi.get('io')}%"
           )
           if tasks:
               t = tasks[0]
               prompt += (
                   f"\n   Top process: {t.get('comm')} "
                   f"(PID {t.get('pid')}, RSS: {t.get('rss_kb')}kB)"
               )

   if oom_events:
       prompt += "\n\nOOM EVENTS:\n"
       for i, ev in enumerate(oom_events[:5], 1):
           ts = ev.get("timestamp_sec", 0)
           if ev.get("kind") == "oom_kill":
               prompt += f"\n{i}. At {ts:.2f}s: OOM kill → {ev.get('task')} (PID {ev.get('pid')})"
           elif ev.get("kind") == "oom_killed_process":
               prompt += f"\n{i}. At {ts:.2f}s: Killed → {ev.get('comm')} (PID {ev.get('pid')})"
           elif ev.get("kind") == "oom_reaper":
               prompt += f"\n{i}. At {ts:.2f}s: OOM reaper reclaimed from {ev.get('comm')}"

   prompt += (
       "\n\nProvide a concise report (300-400 words) covering:\n"
       "1. Overall system health assessment\n"
       "2. Critical issues identified\n"
       "3. Resource pressure trends (CPU, Memory, IO)\n"
       "4. Processes causing the most pressure\n"
       "5. Key recommendations for optimization\n\n"
       "Be specific and actionable."
   )

   try:
       return get_llm_response(prompt, api_key=api_key, model=MODEL)
   except Exception as e:
       return f"Error generating report: {str(e)}"


# ── Chat query ────────────────────────────────────────────────────────────────
def query_logs_with_context(
   query: str,
   store_name: str,
   k: int,
   threshold: float,
   temperature: float,
   json_path: Optional[str] = None,
) -> str:
   api_key = st.session_state.api_key
   embeddings_fn = st.session_state.embeddings_fn
   if not embeddings_fn:
       return "Error: embeddings function not initialised."

   try:
       # Vector retrieval (with built-in threshold fallback in rag.py)
       relevant_docs = query_vector_store(
           store_name=store_name,
           query=query,
           embedding_function=embeddings_fn,
           k=k,
           threshold=threshold,
           base_dir=EMBEDDINGS_DIR,
       )

       # Keyword fallback – deduplicated merge
       if json_path:
           kw_docs = _keyword_fallback_docs(query, json_path, max_events=k)
           existing = {d.page_content for d in relevant_docs}
           for kd in kw_docs:
               if kd.page_content not in existing:
                   relevant_docs.append(kd)
                   existing.add(kd.page_content)

       combined_input = build_combined_input(query, relevant_docs)
       return get_llm_response(
           combined_input,
           api_key=api_key,
           model=MODEL,
           temperature=temperature,
       )
   except Exception as e:
       return f"Error processing query: {str(e)}"


# ── Sidebar – vector store manager ────────────────────────────────────────────
def render_sidebar_stores(k_val, threshold_val, temp_val):
   """
   Render the 'Your Logs' section in the sidebar.
   Each store entry shows:
     • a button to make it the active store
     • ✏️ rename inline
     • 🗑️ delete with confirmation
   """
   stores = list_vector_stores(EMBEDDINGS_DIR)

   st.sidebar.markdown("### 📂 Your Logs")

   if not stores:
       st.sidebar.caption("No saved logs yet. Upload a log file to get started.")
       return

   for sid in stores:
       display = _get_display_name(sid)
       is_active = st.session_state.active_store == sid

       # ── Rename mode ──────────────────────────────────────────────────────
       if st.session_state.rename_target == sid:
           new_name = st.sidebar.text_input(
               "New name",
               value=st.session_state.rename_value or display,
               key=f"rename_input_{sid}",
               label_visibility="collapsed",
           )
           col_ok, col_cancel = st.sidebar.columns([1, 1])
           with col_ok:
               if st.button("✅", key=f"rename_ok_{sid}", use_container_width=True):
                   new_name = new_name.strip()
                   if new_name and new_name != display:
                       _set_display_name(sid, new_name)
                   st.session_state.rename_target = None
                   st.session_state.rename_value = ""
                   st.rerun()
           with col_cancel:
               if st.button("✖", key=f"rename_cancel_{sid}", use_container_width=True):
                   st.session_state.rename_target = None
                   st.session_state.rename_value = ""
                   st.rerun()
           continue

       # ── Delete confirmation ───────────────────────────────────────────────
       if st.session_state.confirm_delete == sid:
           st.sidebar.warning(f"Delete **{display}**?")
           col_yes, col_no = st.sidebar.columns([1, 1])
           with col_yes:
               if st.button("🗑️ Yes", key=f"del_yes_{sid}", use_container_width=True):
                   delete_vector_store(sid, EMBEDDINGS_DIR)
                   _unregister_store(sid)
                   if st.session_state.active_store == sid:
                       st.session_state.active_store = None
                       st.session_state.store_name = None
                       st.session_state.processed = False
                       st.session_state.chat_history = []
                       st.session_state.initial_report = None
                       st.session_state.events_data = None
                   st.session_state.confirm_delete = None
                   st.rerun()
           with col_no:
               if st.button("Cancel", key=f"del_no_{sid}", use_container_width=True):
                   st.session_state.confirm_delete = None
                   st.rerun()
           continue

       # ── Normal row ────────────────────────────────────────────────────────
       col_name, col_edit, col_del = st.sidebar.columns([5, 1, 1])
       with col_name:
           btn_label = f"{'▶️ ' if is_active else ''}{display}"
           if st.button(btn_label, key=f"select_{sid}", use_container_width=True):
               # Switch active store
               st.session_state.active_store = sid
               st.session_state.store_name = sid

               # Load json + events for this store
               meta = _load_store_meta()
               json_path = meta.get(sid, {}).get("json_path")

               if json_path and os.path.exists(json_path):
                  st.session_state.json_path = json_path
                  st.session_state.events_data = _load_events_from_json(json_path)
               else:
                  st.session_state.json_path = None
                  st.session_state.events_data = None

               st.session_state.processed = True

               # Clear chat for the new store
               st.session_state.chat_history = []
               st.session_state.initial_report = meta.get(sid, {}).get("initial_report", "")

               st.rerun()
       with col_edit:
           if st.button("✏️", key=f"edit_{sid}", use_container_width=True, help="Rename"):
               st.session_state.rename_target = sid
               st.session_state.rename_value = display
               st.rerun()
       with col_del:
           if st.button("🗑️", key=f"del_{sid}", use_container_width=True, help="Delete"):
               st.session_state.confirm_delete = sid
               st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
   init_state()

   # ── Sidebar ───────────────────────────────────────────────────────────────
   with st.sidebar:
       # Initialise embeddings silently from .env key - no UI shown to user
       if st.session_state.api_key and st.session_state.embeddings_fn is None:
           try: 
               st.session_state.embeddings_fn = get_embeddings_fn(st.session_state.api_key)
           except Exception as e:
               st.error(f"Count not initialise embeddings: {e}")

       # Retrieval settings
       st.markdown("### 🔍 Retrieval Settings")
       k_value = st.slider(
           "Documents to retrieve (k)",
           1, 20, DEFAULT_K,
           help="More = broader context. Good for intricate queries.",
       )
       threshold_value = st.slider(
           "Similarity threshold",
           0.0, 1.0, DEFAULT_THRESHOLD, 0.01,
           help="Lower = more results. Recommended ≤ 0.1 for specific log queries.",
       )
       temperature_value = st.slider(
           "LLM temperature",
           0.0, 1.0, TEMPERATURE, 0.05,
           help="Lower = more factual/deterministic. Higher = more creative.",
       )

       st.divider()

       # Vector store manager (Your Logs)
       render_sidebar_stores(k_value, threshold_value, temperature_value)

       st.divider()

       # Reset session
       if st.button("🔄 Reset Session", use_container_width=True):
           if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
               try:
                   shutil.rmtree(st.session_state.temp_dir)
               except Exception:
                   pass
           for key in list(st.session_state.keys()):
               del st.session_state[key]
           st.rerun()

   # ── Main area header ──────────────────────────────────────────────────────
   st.markdown("# 📊 PSI-Talk")
   st.caption("Analyze Linux Pressure Stall Information and OOM events with AI-powered insights")

   if not st.session_state.api_key:
       st.error("API_KEY not found in .env - please add it and restart the app")
       st.stop()

   # ── Upload / process ──────────────────────────────────────────────────────
   if not st.session_state.processed:
       st.subheader("📁 Upload Log File")
       st.info("💡 Upload a PSI monitor log file (*.txt or *.log) to begin analysis")

       uploaded_file = st.file_uploader(
           "Choose a file", type=["txt", "log"],
           help="Upload a Linux PSI monitor log file",
       )

       if uploaded_file is not None:
           with st.spinner("🔄 Processing log file…"):
               try:
                   temp_dir, json_path, events_data = process_log_file(uploaded_file)

                   if not events_data.get("events"):
                       st.error("❌ No events found. Please check the file format.")
                       st.stop()

                   store_id = f"psi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                   display_name = uploaded_file.name.replace(".txt", "").replace(".log", "")

                   st.session_state.temp_dir = temp_dir
                   st.session_state.json_path = json_path
                   st.session_state.events_data = events_data
                   st.session_state.store_name = store_id
                   st.session_state.active_store = store_id

                   st.success(
                       f"✅ Processed {len(events_data['events'])} events from {uploaded_file.name}"
                   )
               except Exception as e:
                   st.error(f"❌ Error processing file: {e}")
                   import traceback
                   with st.expander("Show error details"):
                       st.code(traceback.format_exc())
                   st.stop()

           # Metrics preview
           st.subheader("📊 Log Overview")
           display_metrics(st.session_state.events_data)
           st.divider()

           # Build vector store + initial report
           with st.spinner("🤖 Generating AI analysis report…"):
               try:
                   embeddings_fn = st.session_state.embeddings_fn
                   docs = build_rag_docs(json_path)
                   if not docs:
                       st.error("❌ No documents created from the log.")
                       st.stop()

                   vec_store(docs, embeddings_fn, store_id, base_dir=EMBEDDINGS_DIR)

                   # Register store with metadata
                   _register_store(store_id, display_name, uploaded_file.name)
                   # Also persist json_path in metadata for reload
                   meta = _load_store_meta()
                   meta[store_id]["json_path"] = json_path
                   _save_store_meta(meta)

                   report = generate_initial_report(events_data, st.session_state.api_key)

                   # Persist report in metadata
                   meta = _load_store_meta()
                   meta[store_id]["initial_report"] = report
                   _save_store_meta(meta)

                   st.session_state.initial_report = report
                   st.session_state.processed = True
                   st.rerun()

               except Exception as e:
                   st.error(f"❌ Error generating report: {e}")
                   import traceback
                   with st.expander("Show error details"):
                       st.code(traceback.format_exc())
                   st.stop()

   # ── Analyzed view ─────────────────────────────────────────────────────────
   else:
       active_display = (
           _get_display_name(st.session_state.active_store)
           if st.session_state.active_store
           else "Log"
       )
       st.markdown(f"### 📋 AI Report — *{active_display}*")

       if st.session_state.initial_report:
           st.markdown(
               f'<div class="report-box">{st.session_state.initial_report}</div>',
               unsafe_allow_html=True,
           )
                
           # ✅ Build PDF parts
           metrics = compute_metrics_summary(st.session_state.events_data) if st.session_state.events_data else {}
           chart_png = pressure_timeline_png(st.session_state.events_data) if st.session_state.events_data else None

           pdf_bytes = generate_full_pdf_bytes(
                title=f"PSI-Talk AI Analysis - {active_display}",
                report_md=st.session_state.initial_report,
                metrics=metrics,
                chart_png=chart_png,
           )

           st.download_button(
                label="📄 Export AI Analysis to PDF",
                data=pdf_bytes,
                file_name=f"psi_talk_ai_analysis_{st.session_state.active_store or 'log'}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

       else:
           st.info("Select a log from the sidebar or upload a new file.")

       if st.session_state.events_data:
           st.divider()
           st.subheader("📊 Key Metrics")
           display_metrics(st.session_state.events_data)
           st.subheader("📈 Pressure Timeline")
           plot_pressure_over_time(st.session_state.events_data)

       # ── Chat ──────────────────────────────────────────────────────────────
       st.divider()
       st.subheader("💬 Ask Questions About Your Logs")
       st.info(
           "💡 Try: *'When was stress-ng first triggered?'* "
           "or *'Which process caused the OOM kill?'* "
           "or *'What was the CPU pressure at 45 seconds?'*"
       )

       for msg in st.session_state.chat_history:
           with st.chat_message(msg["role"]):
               st.markdown(msg["content"])

       if query := st.chat_input("Ask a question about your PSI logs…"):
           st.session_state.chat_history.append({"role": "user", "content": query})
           with st.chat_message("user"):
               st.markdown(query)

           with st.chat_message("assistant"):
               with st.spinner("Analyzing logs…"):
                   store_name = st.session_state.store_name
                   json_path = st.session_state.json_path

                   if not store_name:
                       response = "❌ No active log store. Please upload a log file first."
                   else:
                       response = query_logs_with_context(
                           query,
                           store_name=store_name,
                           k=k_value,
                           threshold=threshold_value,
                           temperature=temperature_value,
                           json_path=json_path,
                       )

                   st.markdown(response)
                   st.session_state.chat_history.append(
                       {"role": "assistant", "content": response}
                   )


if __name__ == "__main__":
   main()