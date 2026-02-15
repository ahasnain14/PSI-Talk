# parser.py

import json
import re
from typing import Dict, Any, List, Optional

# Made these configurable instead of hardcoded
def parse_log_file(log_file_path: str, output_file_path: str = None):
  """
  Parse a PSI log file and optionally save to JSON.
  
  Args:
      log_file_path: Path to the input log file
      output_file_path: Optional path to save JSON output
  
  Returns:
      Dict containing parsed events
  """
  with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
      lines = f.readlines()
  
  output = parse_log(lines)
  output["events"].sort(key=lambda e: e.get("timestamp_sec", 0.0))
  
  if output_file_path:
      with open(output_file_path, "w") as f:
          json.dump(output, f, indent=2)
      print(f"Wrote {len(output['events'])} events to {output_file_path}")
  
  return output


# Matches leading timestamp 
LEAD_TS = r"\[\s*(?P<ts>\d+\.\d+)\]\s*\[[^\]]*\]\s*"

# pressure high line:
pressure_re = re.compile(
  LEAD_TS
  + r"psi_monitor:\s+pressure high:\s*"
    r"cpu=(?P<cpu>\d+)%\s+mem=(?P<mem>\d+)%\s+io=(?P<io>\d+)%"
    r"(?:\s*\(thresh\s*cpu=(?P<th_cpu>\d+)\s*mem=(?P<th_mem>\d+)\s*io=(?P<th_io>\d+)\))?",
  re.IGNORECASE,
)

# header like: psi_monitor: logging top 10 tasks under pressure:
psi_header_re = re.compile(
  LEAD_TS + r"psi_monitor:\s+logging top\s+\d+\s+tasks\s+under\s+pressure:",
  re.IGNORECASE,
)

# task line:
task_re = re.compile(
  LEAD_TS
  + r"psi_monitor:\s+pid=(?P<pid>\d+)\s+comm=(?P<comm>[A-Za-z0-9._\-]+)\s+"
    r"psi_flag=(?P<psi_flag>\d+)\s+oncpu=(?P<oncpu>\d+)\s+"
    r"cputime\(ms\)=(?P<cpu_ms>\d+)\s+rss\(kB\)=(?P<rss_kb>\d+)\s+"
    r"io\(kB\)=(?P<io_kb>\d+)\s+score=(?P<score>\d+)",
  re.IGNORECASE,
)

# OOM kill line:
oom_kill_re = re.compile(
  LEAD_TS
  + r"oom-kill:(?P<kv>.+)$",
  re.IGNORECASE,
)

# Out of memory summary line:
oom_killed_proc_re = re.compile(
  LEAD_TS
  + r"Out of memory:\s+Killed process\s+(?P<pid>\d+)\s+\((?P<comm>[^)]+)\)\s+(?P<rest>.+)$",
  re.IGNORECASE,
)

# oom_reaper line:
oom_reaper_re = re.compile(
  LEAD_TS
  + r"oom_reaper:\s+reaped process\s+(?P<pid>\d+)\s+\((?P<comm>[^)]+)\),\s+(?P<rest>.+)$",
  re.IGNORECASE,
)

def parse_kv_blob(blob: str) -> Dict[str, Any]:
  """
  Parses comma/space separated key=value blob, tolerates missing '=' segments.
  Example: 'constraint=CONSTRAINT_NONE,nodemask=(null),...,task=stress-ng,pid=4112'
  """
  out: Dict[str, Any] = {}
  parts = [p.strip() for p in blob.split(",")]
  for p in parts:
      if "=" in p:
          k, v = p.split("=", 1)
          out[k.strip()] = v.strip()
      else:
          if p:
              out.setdefault("_extra", []).append(p)
  return out

def to_int_safe(v: Optional[str]) -> Optional[int]:
  try:
      return int(v) if v is not None else None
  except ValueError:
      return None

def make_pressure_event(m: re.Match) -> Dict[str, Any]:
  ev = {
      "kind": "pressure_high",
      "timestamp_sec": float(m.group("ts")),
      "psi": {
          "cpu": int(m.group("cpu")),
          "mem": int(m.group("mem")),
          "io": int(m.group("io")),
      },
      "thresholds": None,
      "tasks": [],
  }
  th_cpu = m.group("th_cpu")
  if th_cpu is not None:
      ev["thresholds"] = {
          "cpu": int(th_cpu),
          "mem": int(m.group("th_mem")),
          "io": int(m.group("th_io")),
      }
  return ev

def make_task(m: re.Match) -> Dict[str, Any]:
  return {
      "pid": int(m.group("pid")),
      "comm": m.group("comm"),
      "psi_flag": int(m.group("psi_flag")),
      "oncpu": int(m.group("oncpu")),
      "cpu_ms": int(m.group("cpu_ms")),
      "rss_kb": int(m.group("rss_kb")),
      "io_kb": int(m.group("io_kb")),
      "score": int(m.group("score")),
      "timestamp_sec": float(m.group("ts")),
  }

def make_oom_kill_event(m: re.Match) -> Dict[str, Any]:
  kv = parse_kv_blob(m.group("kv"))
  return {
      "kind": "oom_kill",
      "timestamp_sec": float(m.group("ts")),
      "details": kv,
      "task": kv.get("task"),
      "pid": to_int_safe(kv.get("pid")),
      "uid": to_int_safe(kv.get("uid")),
      "constraint": kv.get("constraint"),
      "task_memcg": kv.get("task_memcg"),
  }

def make_oom_killed_proc_event(m: re.Match) -> Dict[str, Any]:
  return {
      "kind": "oom_killed_process",
      "timestamp_sec": float(m.group("ts")),
      "pid": int(m.group("pid")),
      "comm": m.group("comm"),
      "summary": m.group("rest"),
  }

def make_oom_reaper_event(m: re.Match) -> Dict[str, Any]:
  return {
      "kind": "oom_reaper",
      "timestamp_sec": float(m.group("ts")),
      "pid": int(m.group("pid")),
      "comm": m.group("comm"),
      "summary": m.group("rest"),
  }

def parse_log(lines: List[str]) -> Dict[str, Any]:
  """
  Parse log lines into structured events.
  
  Args:
      lines: List of log file lines
  
  Returns:
      Dict with 'source' and 'events' keys
  """
  events: List[Dict[str, Any]] = []
  current_event: Optional[Dict[str, Any]] = None

  def flush_current():
      nonlocal current_event
      if current_event:
          events.append(current_event)
          current_event = None

  for line in lines:
      line = line.rstrip("\n")

      # pressure high?
      m = pressure_re.search(line)
      if m:
          flush_current()
          current_event = make_pressure_event(m)
          continue

      # skip header "logging top N" lines
      if psi_header_re.search(line):
          continue

      # task line?
      m = task_re.search(line)
      if m:
          if current_event and current_event.get("kind") == "pressure_high":
              current_event["tasks"].append(make_task(m))
          continue

      # OOMs
      m = oom_kill_re.search(line)
      if m:
          flush_current()
          events.append(make_oom_kill_event(m))
          continue

      m = oom_killed_proc_re.search(line)
      if m:
          flush_current()
          events.append(make_oom_killed_proc_event(m))
          continue

      m = oom_reaper_re.search(line)
      if m:
          flush_current()
          events.append(make_oom_reaper_event(m))
          continue

  if current_event:
      events.append(current_event)

  return {
      "source": "psi_monitor",
      "events": events,
  }

def main():
  """CLI entry point for parsing logs"""
  LOG_FILE = "logs/sa525m-psi-full-scenario-logs.txt"
  OUT_FILE = "data/psi_events.json"
  
  parse_log_file(LOG_FILE, OUT_FILE)

if __name__ == "__main__":
  main()