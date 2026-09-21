"""Public-hardened Agent Reliability Lab v0.3 judge UI.

Preserves the exact R2-certified agent core while constraining the public surface to
local supplied evidence only. Provider secrets remain server-side and are excluded
from browser evidence. Public execution is fail-closed unless explicitly enabled.
"""
from __future__ import annotations

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List
import collections
import html
import json
import os
import re
import threading
import time

from agent_reliability_lab_v0_2 import (
    EvidenceFirstAgent,
    NebiusNemotronAdapter,
    ProviderError,
    TaskManifest,
    ToolMalformed,
    ToolRunner,
    build_report,
)

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


def build_evidence_search(evidence_text: str):
    chunks = [line.strip() for line in evidence_text.splitlines() if line.strip()]

    def search(args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolMalformed("query must be non-empty")
        terms = set(re.findall(r"[a-zA-Z0-9]{2,}", query.lower()))
        scored = []
        for idx, chunk in enumerate(chunks, start=1):
            words = set(re.findall(r"[a-zA-Z0-9]{2,}", chunk.lower()))
            score = len(terms & words)
            if score:
                scored.append((score, idx, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return {
            "value": [
                {"evidence_id": f"E{idx}", "text": chunk, "score": score}
                for score, idx, chunk in scored[:5]
            ]
        }

    return search


def run_demo(payload: Dict[str, Any], *, model_adapter=None) -> Dict[str, Any]:
    goal = str(payload.get("goal") or "").strip()
    evidence = str(payload.get("evidence") or "").strip()
    if not goal:
        raise ValueError("goal is required")
    if len(goal.encode("utf-8")) > 4096:
        raise ValueError("task exceeds 4 KB limit")
    if len(evidence.encode("utf-8")) > 20 * 1024:
        raise ValueError("evidence exceeds 20 KB limit")

    if model_adapter is None:
        model_adapter = NebiusNemotronAdapter(model=DEFAULT_MODEL)

    tools: Dict[str, Any] = {}
    allowed: List[str] = []
    instructions = []

    if evidence:
        tools["evidence_search"] = build_evidence_search(evidence)
        allowed.append("evidence_search")
        instructions.append(
            "Use evidence_search with arguments {\"query\":\"...\"} when the supplied evidence is relevant. "
            "Do not claim that local evidence contains facts you did not retrieve."
        )

    if not allowed:
        instructions.append("No tools are available; answer only from the task context and state uncertainty when needed.")

    manifest = TaskManifest(
        task_id="demo-" + os.urandom(4).hex(),
        goal=goal + "\n\nTool guidance: " + " ".join(instructions),
        allowed_tools=tuple(allowed),
        max_steps=6,
        max_tool_retries=0,
    )
    record = EvidenceFirstAgent(model_adapter, ToolRunner(tools)).run(manifest)
    return {
        "report": build_report(record),
        "events": [asdict(event) for event in record.events],
    }


INDEX_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Reliability Lab</title>
<style>
:root{
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;
  color:#111827;background:#f5f7fb;
  --ink:#111827;--muted:#667085;--line:#e5e7eb;--panel:#ffffff;
  --soft:#f8fafc;--good:#047857;--bad:#b42318;--accent:#2563eb
}
*{box-sizing:border-box}body{margin:0}.wrap{max-width:1180px;margin:0 auto;padding:28px}
.hero{background:linear-gradient(135deg,#111827,#1f2937);color:white;border-radius:20px;padding:28px;box-shadow:0 12px 34px #0002}
.hero h1{margin:0 0 7px;font-size:32px;letter-spacing:-.02em}.hero p{margin:0;color:#d1d5db;max-width:820px}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:18px;font-size:13px;color:#dbeafe}
.flow span{background:#ffffff12;border:1px solid #ffffff24;padding:7px 10px;border-radius:999px}.flow b{color:#93c5fd}
.grid{display:grid;grid-template-columns:1.05fr .95fr;gap:18px;margin-top:18px}
.card{background:var(--panel);border-radius:18px;padding:20px;box-shadow:0 7px 26px #0000000d;border:1px solid var(--line)}
.card h2,.card h3{margin-top:0}
label{font-weight:700;display:block;margin:12px 0 7px}
textarea,input,select{width:100%;padding:12px;border:1px solid #d0d5dd;border-radius:11px;font:inherit;background:white}
textarea{min-height:105px;resize:vertical}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.actions{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}
button{background:var(--ink);color:white;border:0;border-radius:11px;padding:12px 15px;font-weight:750;cursor:pointer}
button.secondary{background:white;color:var(--ink);border:1px solid #d0d5dd}
button:disabled{opacity:.55;cursor:wait}
.muted{color:var(--muted);font-size:14px}.hidden{display:none}
.statusline{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}
.pill{display:inline-flex;align-items:center;gap:7px;border-radius:999px;padding:7px 11px;font-weight:800;font-size:13px}
.pill.pass{background:#ecfdf3;color:var(--good);border:1px solid #a7f3d0}
.pill.fail{background:#fff1f0;color:var(--bad);border:1px solid #fecaca}
.answer{background:#f8fafc;border:1px solid var(--line);border-radius:13px;padding:14px;margin-top:12px;line-height:1.5}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin-top:14px}
.kpi{background:#f9fafb;border:1px solid var(--line);border-radius:12px;padding:10px}.kpi span{font-size:12px;color:var(--muted)}.kpi b{display:block;font-size:21px;margin-top:3px}
.passport{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px}
.check{background:white;border:1px solid var(--line);border-radius:11px;padding:10px;font-size:13px}.check strong{display:block;margin-bottom:3px}
.ok{color:var(--good)}.warn{color:#b54708}
.timeline{margin-top:14px;display:grid;gap:8px}
.event{display:grid;grid-template-columns:32px 1fr;gap:10px;align-items:start;background:#fbfcfe;border:1px solid var(--line);border-radius:12px;padding:10px}
.dot{width:26px;height:26px;border-radius:50%;background:#eaf2ff;color:#1d4ed8;display:grid;place-items:center;font-size:12px;font-weight:800}
.event b{font-size:13px}.event small{display:block;color:var(--muted);margin-top:3px;line-height:1.35}
details{margin-top:14px;border-top:1px solid var(--line);padding-top:12px}
summary{cursor:pointer;font-weight:750;color:#344054}
pre{white-space:pre-wrap;word-break:break-word;background:#0b1020;color:#d1fae5;border-radius:12px;padding:14px;max-height:360px;overflow:auto;font-size:12px}
.note{margin-top:12px;background:#fffaeb;border:1px solid #fedf89;border-radius:11px;padding:10px 12px;color:#7a2e0e;font-size:13px}
@media(max-width:850px){.grid{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}.passport{grid-template-columns:1fr}.row{grid-template-columns:1fr}}
</style></head>
<body><div class="wrap">
<div class="hero">
  <h1>Agent Reliability Lab</h1>
  <p>Make tool-using AI agents show their work: what model ran, what tool was used, what evidence came back, what failed, and whether the workflow stayed inside its boundaries.</p>
  <div class="flow"><span>Task</span><b>→</b><span>Nemotron</span><b>→</b><span>Allowlisted Tool</span><b>→</b><span>Evidence</span><b>→</b><span>Nemotron</span><b>→</b><span>Result + Trace</span></div>
</div>

<div class="grid">
<section class="card">
  <h2>1. Give the agent a bounded task</h2>
  <label for="goal">Task</label>
  <textarea id="goal" placeholder="Example: Based only on the evidence below, determine whether public release is authorized."></textarea>

  <label for="evidence">Local evidence</label>
  <textarea id="evidence" placeholder="One evidence item per line."></textarea>

  <div class="row">
    <div><label>Evidence mode</label><input value="Local supplied evidence only" disabled></div>
    <div><label>Model</label><input value="NVIDIA Nemotron 3 Nano via Nebius Token Factory" disabled></div>
  </div>

  <div class="actions">
    <button id="run">Run bounded agent</button>
    <button id="preset" class="secondary" type="button">Load judge demo</button>
  </div>
  <p class="muted">Credentials stay server-side and are excluded from browser evidence. Public demo mode accepts only supplied local evidence; no web-search provider is enabled.</p>
</section>

<section class="card">
  <div class="statusline"><h2 style="margin:0">2. Reliability result</h2><span id="statusPill" class="pill hidden"></span></div>
  <div id="empty" class="muted" style="margin-top:14px">Run a task to generate a reliability report.</div>
  <div id="result" class="hidden">
    <div class="answer"><strong>Final answer</strong><div id="answer" style="margin-top:6px"></div></div>

    <div class="kpis">
      <div class="kpi"><span>Model calls</span><b id="mc">0</b></div>
      <div class="kpi"><span>Tool calls</span><b id="tc">0</b></div>
      <div class="kpi"><span>Retries</span><b id="rc">0</b></div>
      <div class="kpi"><span>Total tokens</span><b id="tok">0</b></div>
    </div>

    <div class="passport">
      <div class="check"><strong>Allowed-tool boundary</strong><span id="allowCheck">—</span></div>
      <div class="check"><strong>Retry discipline</strong><span id="retryCheck">—</span></div>
      <div class="check"><strong>Final event</strong><span id="finalCheck">—</span></div>
      <div class="check"><strong>Provider evidence</strong><span id="providerCheck">—</span></div>
    </div>
  </div>
</section>
</div>

<section id="traceCard" class="card hidden" style="margin-top:18px">
  <h2>3. What the agent actually did</h2>
  <p class="muted">Human-readable activity timeline derived from the same structured event evidence.</p>
  <div id="timeline" class="timeline"></div>
  <details><summary>Show raw structured evidence</summary><pre id="trace"></pre></details>
</section>

<div class="note">Private candidate UI. A successful private certification applies only to the exact tested source lineage; UI/code changes require fresh review before public certification claims.</div>
</div>

<script>
const $=id=>document.getElementById(id);
const b=$('run');
const labels={
  TASK_START:['1','Task started'],
  MODEL_RESPONSE:['M','Model response'],
  TOOL_REQUEST:['T','Tool requested'],
  TOOL_RESPONSE:['E','Tool evidence returned'],
  RETRY:['R','Bounded retry'],
  FINAL:['✓','Final answer'],
  TASK_END:['2','Task ended'],
  RUN_FAIL:['!','Run failed']
};
function eventSummary(e){
  const p=e.payload||{};
  if(e.event_type==='MODEL_RESPONSE') return `${p.provider||'provider'} · ${p.model||'model'} · finish=${p.finish_reason||'—'} · ${p.total_tokens||0} tokens`;
  if(e.event_type==='TOOL_REQUEST') return `${p.name||'tool'} requested`;
  if(e.event_type==='TOOL_RESPONSE') return `${p.name||'tool'} returned evidence`;
  if(e.event_type==='FINAL') return 'Agent emitted a contract-conforming final action';
  if(e.event_type==='RUN_FAIL') return `${p.class||'failure'}: ${p.error||'run stopped'}`;
  return e.event_type||'event';
}
function renderTimeline(events){
  $('timeline').innerHTML='';
  events.forEach(e=>{
    const [icon,title]=labels[e.event_type]||['•',e.event_type||'Event'];
    const div=document.createElement('div');div.className='event';
    const dot=document.createElement('div');dot.className='dot';dot.textContent=icon;
    const body=document.createElement('div');
    const strong=document.createElement('b');strong.textContent=title;
    const small=document.createElement('small');small.textContent=eventSummary(e);
    body.append(strong,small);div.append(dot,body);$('timeline').append(div);
  });
}
$('preset').onclick=()=>{
  $('goal').value='Based only on the supplied policy evidence, determine whether public release is authorized. If the evidence does not support release, say so.';
  $('evidence').value='Release status: private testing approved.\nPublic publication requires separate owner approval.\nNo public-release approval has been issued.';
};
b.onclick=async()=>{
  b.disabled=true;b.textContent='Running…';
  $('empty').classList.add('hidden');$('result').classList.remove('hidden');$('traceCard').classList.remove('hidden');
  try{
    const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({goal:$('goal').value,evidence:$('evidence').value})});
    const d=await r.json();if(!r.ok)throw new Error(d.error||'request failed');
    const p=d.report, ev=d.events||[];
    $('statusPill').classList.remove('hidden','pass','fail');
    $('statusPill').textContent=p.status+(p.failure_class?' · '+p.failure_class:'');
    $('statusPill').classList.add(p.status==='PASS'?'pass':'fail');
    $('answer').textContent=p.final_output||'(no final visible output)';
    $('mc').textContent=p.model_calls;$('tc').textContent=p.tool_calls;$('rc').textContent=p.retries_used;$('tok').textContent=p.total_tokens;
    const reqs=ev.filter(x=>x.event_type==='TOOL_REQUEST');
    const declared=reqs.every(x=>(x.payload||{}).name==='evidence_search');
    $('allowCheck').textContent=declared?'PASS — declared tools only':'CHECK';
    $('allowCheck').className=declared?'ok':'warn';
    $('retryCheck').textContent=p.retries_used===0?'PASS — no retries':`Observed: ${p.retries_used}`;
    $('retryCheck').className=p.retries_used===0?'ok':'warn';
    const finalPresent=ev.some(x=>x.event_type==='FINAL');
    $('finalCheck').textContent=finalPresent?'PASS — final event present':'FAIL / unavailable';
    $('finalCheck').className=finalPresent?'ok':'warn';
    const modelEvent=ev.find(x=>x.event_type==='MODEL_RESPONSE');
    $('providerCheck').textContent=modelEvent?`${modelEvent.payload.provider||'provider'} / ${modelEvent.payload.model||'model'}`:'No model evidence';
    $('providerCheck').className=modelEvent?'ok':'warn';
    renderTimeline(ev);$('trace').textContent=JSON.stringify(ev,null,2);
  }catch(e){
    $('statusPill').classList.remove('hidden','pass');$('statusPill').classList.add('fail');$('statusPill').textContent='ERROR';
    $('answer').textContent=e.message;$('timeline').innerHTML='';$('trace').textContent='';
  }finally{b.disabled=false;b.textContent='Run bounded agent'}
};
</script></body></html>'''


PUBLIC_ENABLED = os.getenv("ARL_PUBLIC_DEMO_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
TRUST_PROXY = os.getenv("ARL_TRUST_PROXY", "true").strip().lower() in {"1", "true", "yes"}
PER_IP_RUNS_PER_HOUR = max(1, int(os.getenv("ARL_PER_IP_RUNS_PER_HOUR", "3")))
DAILY_RUN_LIMIT = max(1, int(os.getenv("ARL_DAILY_RUN_LIMIT", "20")))
MAX_BODY_BYTES = 32 * 1024
_ACTIVE = threading.BoundedSemaphore(value=1)
_RATE_LOCK = threading.Lock()
_IP_RUNS: Dict[str, collections.deque] = {}
_DAILY_STATE = {"day": None, "accepted": 0}


def _client_ip(handler: BaseHTTPRequestHandler) -> str:
    if TRUST_PROXY:
        xff = handler.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if xff:
            return xff[:128]
    return str(handler.client_address[0])[:128]


def _reserve_run(ip: str, now: float | None = None) -> tuple[bool, str | None]:
    now = time.time() if now is None else now
    hour_ago = now - 3600
    utc_day = time.strftime("%Y-%m-%d", time.gmtime(now))
    with _RATE_LOCK:
        if _DAILY_STATE["day"] != utc_day:
            _DAILY_STATE["day"] = utc_day
            _DAILY_STATE["accepted"] = 0
        if _DAILY_STATE["accepted"] >= DAILY_RUN_LIMIT:
            return False, "demo budget reached"
        q = _IP_RUNS.setdefault(ip, collections.deque())
        while q and q[0] < hour_ago:
            q.popleft()
        if len(q) >= PER_IP_RUNS_PER_HOUR:
            return False, "rate limit reached"
        q.append(now)
        _DAILY_STATE["accepted"] += 1
        return True, None


class Handler(BaseHTTPRequestHandler):
    server_version = "AgentReliabilityLab/0.3-public"

    def _headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self.send_header("Cache-Control", "no-store")

    def _json(self, status: int, obj: Any):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self._headers()
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/healthz":
            self._json(200, {"ok": True, "provider_execution_enabled": PUBLIC_ENABLED})
            return
        if path != "/":
            self.send_error(404)
            return
        raw = INDEX_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self._headers()
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/run":
            self.send_error(404)
            return
        if not PUBLIC_ENABLED:
            self._json(503, {"error": "demo execution disabled"})
            return
        if not _ACTIVE.acquire(blocking=False):
            self._json(429, {"error": "demo busy"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request must be a JSON object")
            if set(payload) - {"goal", "evidence"}:
                raise ValueError("unsupported request fields")
            ok, why = _reserve_run(_client_ip(self))
            if not ok:
                self._json(429 if why == "rate limit reached" else 503, {"error": why})
                return
            self._json(200, run_demo(payload))
        except (ValueError, ProviderError, ToolMalformed) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(500, {"error": type(exc).__name__})
        finally:
            _ACTIVE.release()

    def log_message(self, fmt, *args):
        return


def main():
    host = os.getenv("ARL_HOST") or ("0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    port = int(os.getenv("ARL_PORT") or os.getenv("PORT") or "8765")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Agent Reliability Lab v0.3 public-hardened running at http://{host}:{port}; provider execution enabled={PUBLIC_ENABLED}")
    server.serve_forever()


if __name__ == "__main__":
    main()
