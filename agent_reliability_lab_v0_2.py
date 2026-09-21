"""Agent Reliability Lab v0.2

Public-safe clean-room core for evaluating bounded, tool-using AI agents.
Provider adapters are replaceable. Secrets are never embedded in source.

This module is intentionally dependency-light and uses Python's standard library.
External provider calls occur only when an explicit transport is supplied and invoked.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid


class ReliabilityError(RuntimeError):
    """Base class for fail-closed reliability errors."""


class ProviderError(ReliabilityError):
    pass


class ProviderMalformed(ReliabilityError):
    pass


class ToolTimeout(ReliabilityError):
    pass


class ToolMalformed(ReliabilityError):
    pass


class UndeclaredTool(ReliabilityError):
    pass


class StepLimitExceeded(ReliabilityError):
    pass


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ModelResponse:
    provider: str
    model: str
    response_id: Optional[str]
    content: Optional[str]
    reasoning: Optional[str]
    finish_reason: Optional[str]
    usage: TokenUsage
    latency_ms: int


@dataclass(frozen=True)
class ModelRequest:
    messages: Tuple[Dict[str, str], ...]
    max_tokens: int = 256
    temperature: float = 0.0


class ModelAdapter(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse:
        ...


class JsonTransport(Protocol):
    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Tuple[Mapping[str, Any], int]:
        ...


class UrllibJsonTransport:
    """Small stdlib HTTPS JSON transport.

    It is inert until post_json is called. No credentials are stored here.
    """

    def post_json(
        self,
        url: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_s: float,
    ) -> Tuple[Mapping[str, Any], int]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        for key, value in headers.items():
            req.add_header(key, value)
        req.add_header("Content-Type", "application/json")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(str(exc)) from exc
        latency_ms = int((time.perf_counter() - started) * 1000)
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderMalformed("provider returned non-JSON response") from exc
        if not isinstance(decoded, dict):
            raise ProviderMalformed("provider response must be a JSON object")
        return decoded, latency_ms


class NebiusNemotronAdapter:
    """OpenAI-compatible Nebius Token Factory adapter.

    The API key is injected at construction time or loaded from an environment
    variable. The adapter never writes the secret into evidence records.
    """

    DEFAULT_ENDPOINT = "https://api.tokenfactory.nebius.com/v1/chat/completions"

    def __init__(
        self,
        *,
        model: str,
        api_key: Optional[str] = None,
        api_key_env: str = "NEBIUS_API_KEY",
        endpoint: str = DEFAULT_ENDPOINT,
        transport: Optional[JsonTransport] = None,
        timeout_s: float = 45.0,
    ) -> None:
        key = (api_key if api_key is not None else os.getenv(api_key_env, "")).strip()
        if not key:
            raise ProviderError(f"missing API key; set {api_key_env}")
        self._api_key = key
        self.model = model
        self.endpoint = endpoint
        self.transport = transport or UrllibJsonTransport()
        self.timeout_s = timeout_s

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": list(request.messages),
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": False,
        }
        data, latency_ms = self.transport.post_json(
            self.endpoint,
            {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"},
            payload,
            self.timeout_s,
        )
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderMalformed("missing choices[0].message") from exc
        usage_raw = data.get("usage") or {}
        usage = TokenUsage(
            int(usage_raw.get("prompt_tokens") or 0),
            int(usage_raw.get("completion_tokens") or 0),
            int(usage_raw.get("total_tokens") or 0),
        )
        return ModelResponse(
            provider="Nebius Token Factory",
            model=str(data.get("model") or self.model),
            response_id=data.get("id"),
            content=message.get("content"),
            reasoning=message.get("reasoning"),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
            latency_ms=latency_ms,
        )


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class TaskManifest:
    task_id: str
    goal: str
    allowed_tools: Tuple[str, ...]
    max_steps: int = 6
    max_tool_retries: int = 1
    expected_final: Optional[str] = None


@dataclass
class EvidenceEvent:
    seq: int
    event_type: str
    payload: Dict[str, Any]


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    started_at: float
    finished_at: float
    status: str
    failure_class: Optional[str]
    final_output: Optional[str]
    events: List[EvidenceEvent] = field(default_factory=list)
    model_calls: int = 0
    tool_calls: int = 0
    retries_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def canonical_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("run_id", None)
        d.pop("started_at", None)
        d.pop("finished_at", None)
        return d

    def canonical_digest(self) -> str:
        blob = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ToolRunner:
    def __init__(self, tools: Mapping[str, Callable[[Dict[str, Any]], Dict[str, Any]]]):
        self.tools = dict(tools)

    def call(self, manifest: TaskManifest, call: ToolCall) -> Dict[str, Any]:
        if call.name not in manifest.allowed_tools or call.name not in self.tools:
            raise UndeclaredTool(call.name)
        out = self.tools[call.name](call.arguments)
        if not isinstance(out, dict) or "value" not in out:
            raise ToolMalformed(f"tool={call.name} returned invalid schema")
        return out


class EvidenceFirstAgent:
    """Bounded ReAct-style agent with strict JSON action contract.

    Every model and tool interaction is logged without recording provider secrets.
    The model may emit only one of:
      {"action":"tool","tool":"name","arguments":{...}}
      {"action":"final","answer":"..."}
    """

    SYSTEM_PROMPT = (
        "You are a bounded evidence-first agent. Return exactly one JSON object and no markdown. "
        "Choose either a tool action or a final answer. Never name a tool outside the allowed list. "
        "Tool action schema: {\"action\":\"tool\",\"tool\":\"name\",\"arguments\":{}}. "
        "Final schema: {\"action\":\"final\",\"answer\":\"text\"}."
    )

    def __init__(self, model: ModelAdapter, runner: ToolRunner):
        self.model = model
        self.runner = runner

    @staticmethod
    def _decode_action(content: Optional[str]) -> Dict[str, Any]:
        if not content:
            raise ProviderMalformed("model returned empty visible content")
        try:
            obj = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderMalformed("model did not return valid JSON action") from exc
        if not isinstance(obj, dict) or obj.get("action") not in {"tool", "final"}:
            raise ProviderMalformed("model action must be tool or final")
        return obj

    def run(self, manifest: TaskManifest) -> RunRecord:
        started = time.time()
        events: List[EvidenceEvent] = []
        seq = 0
        final_output: Optional[str] = None
        status = "PASS"
        failure_class: Optional[str] = None
        model_calls = tool_calls = retries_used = 0
        prompt_tokens = completion_tokens = total_tokens = 0
        transcript: List[Dict[str, str]] = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "goal": manifest.goal,
                        "allowed_tools": list(manifest.allowed_tools),
                        "max_steps": manifest.max_steps,
                    },
                    separators=(",", ":"),
                ),
            },
        ]

        def log(kind: str, payload: Dict[str, Any]) -> None:
            nonlocal seq
            seq += 1
            events.append(EvidenceEvent(seq, kind, payload))

        log("TASK_START", {"task_id": manifest.task_id, "goal": manifest.goal})
        try:
            for step_no in range(1, manifest.max_steps + 1):
                response = self.model.complete(
                    ModelRequest(messages=tuple(transcript), max_tokens=1024, temperature=0.0)
                )
                model_calls += 1
                prompt_tokens += response.usage.prompt_tokens
                completion_tokens += response.usage.completion_tokens
                total_tokens += response.usage.total_tokens
                log(
                    "MODEL_RESPONSE",
                    {
                        "step": step_no,
                        "provider": response.provider,
                        "model": response.model,
                        "response_id": response.response_id,
                        "finish_reason": response.finish_reason,
                        "usage": asdict(response.usage),
                        "latency_ms": response.latency_ms,
                        "has_reasoning": bool(response.reasoning),
                    },
                )
                if response.finish_reason == "length":
                    raise ProviderMalformed("model output truncated at completion token ceiling")
                action = self._decode_action(response.content)
                transcript.append({"role": "assistant", "content": response.content or ""})

                if action["action"] == "final":
                    answer = action.get("answer")
                    if not isinstance(answer, str):
                        raise ProviderMalformed("final answer must be a string")
                    final_output = answer
                    log("FINAL", {"answer": answer})
                    break

                tool_name = action.get("tool")
                arguments = action.get("arguments")
                if not isinstance(tool_name, str) or not isinstance(arguments, dict):
                    raise ProviderMalformed("tool action missing tool/arguments")
                call = ToolCall(tool_name, arguments)
                log("TOOL_REQUEST", {"name": tool_name, "arguments": arguments})
                attempt = 0
                while True:
                    try:
                        out = self.runner.call(manifest, call)
                        tool_calls += 1
                        log("TOOL_RESPONSE", {"name": tool_name, "output": out, "attempt": attempt + 1})
                        transcript.append(
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {"tool_result": {"name": tool_name, "output": out}},
                                    separators=(",", ":"),
                                ),
                            }
                        )
                        break
                    except ToolTimeout as exc:
                        log("TOOL_TIMEOUT", {"name": tool_name, "attempt": attempt + 1, "error": str(exc)})
                        if attempt >= manifest.max_tool_retries:
                            raise
                        attempt += 1
                        retries_used += 1
            else:
                raise StepLimitExceeded(f"max_steps={manifest.max_steps}")

            if final_output is None:
                raise StepLimitExceeded("no final answer before step limit")
            if manifest.expected_final is not None and final_output != manifest.expected_final:
                status = "FAIL"
                failure_class = "FINAL_ASSERTION_MISMATCH"
                log("ASSERTION_FAIL", {"expected": manifest.expected_final, "actual": final_output})
        except UndeclaredTool as exc:
            status, failure_class = "FAIL_CLOSED", "UNDECLARED_TOOL"
            log("RUN_FAIL", {"class": failure_class, "error": str(exc)})
        except ToolMalformed as exc:
            status, failure_class = "FAIL_CLOSED", "MALFORMED_TOOL_RESPONSE"
            log("RUN_FAIL", {"class": failure_class, "error": str(exc)})
        except ToolTimeout as exc:
            status, failure_class = "FAIL_CLOSED", "TOOL_TIMEOUT_RETRIES_EXHAUSTED"
            log("RUN_FAIL", {"class": failure_class, "error": str(exc)})
        except ProviderMalformed as exc:
            status, failure_class = "FAIL_CLOSED", "MALFORMED_MODEL_RESPONSE"
            log("RUN_FAIL", {"class": failure_class, "error": str(exc)})
        except ProviderError as exc:
            status, failure_class = "FAIL_CLOSED", "PROVIDER_ERROR"
            log("RUN_FAIL", {"class": failure_class, "error": str(exc)})
        except StepLimitExceeded as exc:
            status, failure_class = "FAIL_CLOSED", "STEP_LIMIT_EXCEEDED"
            log("RUN_FAIL", {"class": failure_class, "error": str(exc)})

        log("TASK_END", {"status": status, "failure_class": failure_class})
        return RunRecord(
            run_id=str(uuid.uuid4()),
            task_id=manifest.task_id,
            started_at=started,
            finished_at=time.time(),
            status=status,
            failure_class=failure_class,
            final_output=final_output,
            events=events,
            model_calls=model_calls,
            tool_calls=tool_calls,
            retries_used=retries_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


def build_report(record: RunRecord) -> Dict[str, Any]:
    return {
        "task_id": record.task_id,
        "status": record.status,
        "failure_class": record.failure_class,
        "final_output": record.final_output,
        "model_calls": record.model_calls,
        "tool_calls": record.tool_calls,
        "retries_used": record.retries_used,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        "total_tokens": record.total_tokens,
        "event_count": len(record.events),
        "run_digest": record.canonical_digest(),
    }


class TavilySearchTool:
    """Optional public-web search adapter for a later, separately enabled path.

    This class exists so the core stays provider-neutral. It does not activate
    unless called and requires a key supplied at runtime.
    """

    ENDPOINT = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None, api_key_env: str = "TAVILY_API_KEY") -> None:
        self.api_key = (api_key if api_key is not None else os.getenv(api_key_env, "")).strip()
        if not self.api_key:
            raise ProviderError(f"missing API key; set {api_key_env}")

    def __call__(self, args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ToolMalformed("search query must be non-empty")
        payload = json.dumps(
            {"api_key": self.api_key, "query": query.strip(), "search_depth": "basic", "max_results": 5}
        ).encode("utf-8")
        req = urllib.request.Request(self.ENDPOINT, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ToolTimeout(str(exc)) from exc
        except json.JSONDecodeError as exc:
            raise ToolMalformed("search provider returned non-JSON") from exc
        results = data.get("results")
        if not isinstance(results, list):
            raise ToolMalformed("search provider missing results")
        safe_results = [
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
                "score": item.get("score"),
            }
            for item in results[:5]
            if isinstance(item, dict)
        ]
        return {"value": safe_results}
