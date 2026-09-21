# Agent Reliability Lab

**Make tool-using AI agents show their work.**

Agent Reliability Lab is a bounded, evidence-first agent runner that exposes what a tool-using workflow actually did: model calls, tool calls, retries, failures, token usage, final outcome, and the execution trace behind it.

The public demo uses **NVIDIA Nemotron 3 Nano through Nebius Token Factory** and a single allowlisted local-evidence tool. The core is intentionally small: reliability evidence is part of the product, not a hidden debugging afterthought.

## What it demonstrates

- bounded multi-step agent execution
- strict JSON action contracts
- allowlisted tool execution
- hard step ceilings and zero tool retries in the demo path
- provider, model, response ID, finish reason, latency, and token evidence
- malformed output and undeclared-tool fail-closed behavior
- server-side credential handling
- browser-visible reliability passport and execution timeline
- provider-neutral model/tool adapter boundaries

## Architecture

```text
Task + supplied evidence
        |
        v
Bounded evidence-first agent
        |
        +--> Nebius Token Factory / NVIDIA Nemotron
        |
        +--> allowlisted evidence_search tool
        |
        v
Reliability evidence ledger
        |
        v
Result + calls + retries + tokens + event trace
```

The hackathon runtime is Nebius Token Factory with:

```text
nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
```

The model adapter is replaceable. Provider credentials are supplied at runtime and are excluded from the browser response and evidence report.

## Requirements

- Python 3.11+
- a Nebius Token Factory API key in `NEBIUS_API_KEY`

The application uses only the Python standard library. No third-party Python package is required.

## Run tests

```bash
python -m unittest -v \
  test_agent_reliability_lab_v0_2.py \
  test_agent_reliability_lab_app_v0_3_public.py
```

The public-hardened candidate has **28/28 deterministic offline tests passing**.

## Run locally

Set the API key in your environment, then enable execution:

```bash
export NEBIUS_API_KEY="..."
export ARL_PUBLIC_DEMO_ENABLED=true
python agent_reliability_lab_app_v0_3_public.py
```

On Windows PowerShell:

```powershell
$env:NEBIUS_API_KEY="..."
$env:ARL_PUBLIC_DEMO_ENABLED="true"
python agent_reliability_lab_app_v0_3_public.py
```

Open:

```text
http://127.0.0.1:8765
```

## Public-demo boundaries

The public surface is deliberately narrower than the private engineering build:

- supplied local evidence only
- no Tavily or arbitrary web fetch
- fixed Nemotron model in the public UI
- maximum 6 model steps
- 0 tool retries
- maximum 20 KB evidence payload
- maximum 4 KB task payload
- maximum 32 KB request body
- one active provider execution at a time
- configurable per-IP hourly limit, default 3
- configurable global daily run ceiling, default 20
- environment-controlled kill switch: `ARL_PUBLIC_DEMO_ENABLED`
- no provider fallback and no automatic model switching

If a provider or boundary fails, the run fails visibly rather than manufacturing success.

## Environment controls

| Variable | Purpose | Default |
| --- | --- | --- |
| `NEBIUS_API_KEY` | Server-side Token Factory credential | unset |
| `ARL_PUBLIC_DEMO_ENABLED` | Provider execution kill switch | `false` |
| `ARL_HOST` | Bind host | `127.0.0.1` |
| `ARL_PORT` | Local port; hosting platforms may supply `PORT` instead | `8765` |
| `ARL_PER_IP_RUNS_PER_HOUR` | Per-IP execution ceiling | `3` |
| `ARL_DAILY_RUN_LIMIT` | Process-level daily accepted-run ceiling | `20` |
| `ARL_TRUST_PROXY` | Honor first `X-Forwarded-For` hop | `true` |

## Reliability lineage

The agent core is the exact R2-certified core used in the successful private end-to-end Nebius/Nemotron run. That private certification demonstrated two model calls, one allowlisted evidence-search tool call, zero retries, positive token accounting, response IDs, `stop` finish reasons, and a contract-conforming final answer.

The public-hardened v0.3 UI preserves that core while adding judge-facing comprehension and public-abuse controls. Public release is conditioned on the separately recorded v0.3 live-conformance gate matching the released source hashes.

## Privacy and logging

The application does not intentionally persist API keys, authorization headers, full prompts, or evidence. The built-in server does not write access logs. Hosting-provider infrastructure may still produce network/platform telemetry according to that provider's own policies.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
