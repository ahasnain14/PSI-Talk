# PSI GenAI Monitor & Analysis Tool

An AI-powered diagnostic assistant for Linux Pressure Stall Information (PSI) monitoring and analysis. This project combines in-kernel PSI monitoring with GenAI-powered insights to help diagnose performance issues in embedded automotive systems.

## 🎯 Overview

This tool bridges the gap between kernel-level PSI metrics and human-readable performance insights. It collects real-time pressure data from the Linux kernel, processes it through a RAG (Retrieval-Augmented Generation) pipeline, and uses AI to provide actionable recommendations for system tuning and debugging.

**Primary Use Case:** Qualcomm Automotive Telematics - In-Kernel PSI Enhancements

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Kernel Space (Linux 5.15+)                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐   │
│  │ psi_monitor.c│───▶️   sysfs knobs   ───▶️│  tracepoints │   │
│  │  (workqueue) │      │              │      │  (optional)  │   │
│  └──────────────┘      └──────────────┘      └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼ dmesg logs
┌─────────────────────────────────────────────────────────────────┐
│                        User Space (Python)                      │
│  ┌──────────────┐      ┌──────────────┐    ┌──────────────┐     │
│  │   parser.py  │───▶️  rag_loader.py ───▶️│embedder.py  │     │
│  │  (log parse) │      │(doc builder) │    │ (vector DB)  │     │
│  └──────────────┘      └──────────────┘    └──────────────┘     │
│                              │                                  │
│                              ▼                                  │
│                      ┌──────────────┐                           │
│                      │   rag.py     │◀️─── QGenie API          |
│                      │ (RAG engine) │     (Claude 4.5)          │
│                      └──────────────┘                           │
│                              │                                  │
│                              ▼                                  │
│                      ┌──────────────┐                           │
│                      │   app.py     │                           │
│                      │  (Streamlit) │                           │
│                      └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow

1. **Kernel Detection** → Kernel detects PSI threshold breach
2. **Task Logging** → Logs top-N tasks with metrics (PID, comm, CPU ms, RSS kB, I/O kB, score)
3. **Log Parsing** → Python extracts PSI blocks from dmesg
4. **JSON Serialization** → Creates structured snapshots with PSI metrics, thresholds, and task data
5. **RAG Processing** → Converts events to human-readable documents and embeds them
6. **AI Analysis** → Queries GenAI model for insights and recommendations
7. **User Interface** → Streamlit app presents analysis results

## 🚀 Features

- **Real-time PSI Monitoring**: Captures CPU, memory, and I/O pressure events
- **OOM Event Tracking**: Monitors out-of-memory kills and reaper activity
- **Intelligent Parsing**: Extracts structured data from kernel logs
- **Vector Search**: RAG-based retrieval for relevant historical events
- **AI-Powered Analysis**: Natural language insights using Claude 4.5 Sonnet
- **Interactive UI**: Streamlit-based interface for querying and visualization
- **Keyword Fallback**: Ensures relevant results even for specific queries

## 📋 Prerequisites

- Python 3.8+
- Linux system with PSI support (5.15+ recommended)
- QGenie API key (for embeddings and LLM)
- Kernel logs with PSI monitor output

## 🔧 Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd PSI-Talk
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
export API_KEY="your-qgenie-api-key-here"
```

## 📁 Project Structure

```
.
├── app.py                      # Streamlit web application
├── src/
│   ├── parser.py              # Log file parser (PSI + OOM events)
│   ├── rag_loader.py          # Document builder for RAG
│   ├── embedder.py            # Vector store management
│   └── rag.py                 # RAG query engine
├── data/
│   └── psi_events.json        # Parsed events (generated)
├── logs/
│   └── *.txt                  # Kernel log files
├── embeddings/                 # Vector database storage
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🎮 Usage (Local)

### 1. Parse Kernel Logs

```bash
python -m src.parser
```

This reads from `logs/*_.txt` and outputs structured JSON to `data/psi_events.json`.

**Custom paths:**
```python
from src.parser import parse_log_file

parse_log_file(
   log_file_path="path/to/your/log.txt",
   output_file_path="path/to/output.json"
)
```

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

The web interface allows you to:
- Ask natural language questions about PSI events
- View retrieved relevant documents
- Get AI-powered analysis and recommendations
- Explore parsed event data

### 3. Programmatic Query

```python
from src.rag import run_query
import os

api_key = os.getenv("API_KEY")

response = run_query(
   query="What caused the highest memory pressure?",
   api_key=api_key,
   k=8,  # Number of documents to retrieve
   threshold=0.05,  # Similarity threshold
   temperature=0.3,  # LLM temperature
   verbose=True
)

print(response)
```

## 📝 Event Types

The parser recognizes the following event types:

| Event Type | Description | Key Fields |
|------------|-------------|------------|
| `pressure_high` | PSI threshold breach | cpu%, mem%, io%, tasks list |
| `oom_kill` | OOM killer invoked | task, pid, constraint |
| `oom_killed_process` | Process killed by OOM | pid, comm, summary |
| `oom_reaper` | OOM reaper reclaimed memory | pid, comm, summary |

## 🔍 Example Queries

- "What process consumed the most memory during the test?"
- "When did the highest I/O pressure occur?"
- "Which tasks were killed by the OOM killer?"
- "What was the CPU pressure at timestamp 1234.567?"
- "Analyze the pressure patterns and recommend tuning steps"

## ⚙️ Configuration

### RAG Parameters (in `src/rag.py`)

```python
DEFAULT_K = 8                  # Number of documents to retrieve
DEFAULT_THRESHOLD = 0.05       # Similarity score threshold
TEMPERATURE = 0.3              # LLM sampling temperature
MAX_TOKENS = 2048              # Maximum response length
```

### Model Configuration

```python
MODEL = "anthropic::claude-4-5-sonnet"
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 1.1
```

## 🧪 Testing

### Test Parser
```bash
python -m src.parser
# Check data/psi_events.json for structured output
```

### Test RAG Loader
```bash
python -m src.rag_loader
# Should output: "Built N RAG docs"
```

### Test Embeddings
```python
from src.embedder import vec_store, get_embeddings_fn
from src.rag_loader import build_rag_docs
import os

api_key = os.getenv("QGENIE_API_KEY")
docs = build_rag_docs("data/psi_events.json")
embeddings_fn = get_embeddings_fn(api_key)
db = vec_store(docs, embeddings_fn, "test_db")
print(f"Created vector store with {len(docs)} documents")
```

## 🎯 Advanced Features

### Keyword Fallback

When vector similarity search returns no results, the system automatically falls back to keyword-based search over raw events. This ensures specific queries (e.g., "process with PID 1234") always return relevant context.

### Hybrid Retrieval

The RAG pipeline combines:
1. Vector similarity search (primary)
2. Keyword-based fallback (secondary)
3. Deduplication to avoid redundant context

### Timestamp-Aware Analysis

The AI model is specifically prompted to:
- Extract exact timestamps from events
- Quote specific numeric values
- Provide time-ordered analysis
- Avoid vague "I'm not sure" responses

## 📊 Sample Output

**Query:** "What caused the memory pressure spike?"

**AI Response:**
```
At timestamp 1234.567890 seconds, the system experienced severe memory pressure 
at 87%. The top contributing process was navd (pid 4521) which consumed 
245,678 kB of RAM and 1,234 ms of CPU time. 

The pressure event also logged map_renderer (pid 4522) with 198,432 kB RSS, 
suggesting concurrent memory-intensive operations.

Recommendations:
1. Lower mem_thresh to 75% for earlier detection
2. Consider memory limits on navd via cgroups
3. Investigate map_renderer for potential memory leaks
```

## 🛠️ Troubleshooting

### No documents retrieved
- Check that `data/psi_events.json` exists and contains events
- Verify vector store was created: `ls embeddings/psi_vector_db/`
- Try lowering the similarity threshold (default: 0.05)

### API key errors
- Ensure `QGENIE_API_KEY` environment variable is set
- Verify the API key is valid

### Parser issues
- Check log file format matches expected PSI monitor output
- Verify timestamps are in `[seconds.microseconds]` format

## 🔐 Security Notes

- **Never commit API keys** to version control
- Use `.env` files for local development
- Anonymize data before sharing logs externally
- Do not run AI API calls inside kernel space

## 🚧 Future Enhancements

- [ ] Live tracing mode using kernel tracepoints
- [ ] Per-cgroup pressure analysis
- [ ] On-device model execution
- [ ] Automated tuning via feedback loops
- [ ] Real-time dashboard with live updates
- [ ] Historical trend analysis and visualization
- [ ] Integration with perf/trace-cmd/eBPF tools

## 📚 References

- [Linux PSI Documentation](https://www.kernel.org/doc/html/latest/accounting/psi.html)
- [QGenie API Documentation](https://qgenie-sdk-python.qualcomm.com/qgenie_sdk_core/index.html)
- [LangChain Documentation](https://python.langchain.com/)
- [Chromadb Documentation](https://docs.trychroma.com/)


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Note**: This tool is designed for debugging and profiling embedded automotive systems. Always test thoroughly in non-production environments before deploying to production systems.

