import json
import unittest

from agent_reliability_lab_v0_2 import (
    EvidenceFirstAgent,
    ModelRequest,
    ModelResponse,
    NebiusNemotronAdapter,
    ProviderError,
    TaskManifest,
    TokenUsage,
    ToolRunner,
    ToolTimeout,
    build_report,
)


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, request: ModelRequest):
        self.calls += 1
        content = self.outputs.pop(0)
        return ModelResponse(
            provider="synthetic",
            model="fixture-model",
            response_id=f"r{self.calls}",
            content=content,
            reasoning=None,
            finish_reason="stop",
            usage=TokenUsage(10, 5, 15),
            latency_ms=1,
        )


class FixtureTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post_json(self, url, headers, payload, timeout_s):
        self.calls.append((url, dict(headers), dict(payload), timeout_s))
        return self.response, 7


class CounterTool:
    def __init__(self, failures=0, malformed=False):
        self.calls = 0
        self.failures = failures
        self.malformed = malformed

    def __call__(self, args):
        self.calls += 1
        if self.calls <= self.failures:
            raise ToolTimeout("synthetic timeout")
        if self.malformed:
            return {"bad": True}
        return {"value": args.get("value")}


class TestV02(unittest.TestCase):
    def manifest(self, **kwargs):
        data = dict(task_id="t1", goal="verify alpha", allowed_tools=("lookup",), max_steps=4, max_tool_retries=1)
        data.update(kwargs)
        return TaskManifest(**data)

    def agent(self, outputs, tool=None):
        return EvidenceFirstAgent(ScriptedModel(outputs), ToolRunner({"lookup": tool or CounterTool()}))

    def test_01_direct_final_pass(self):
        rec = self.agent(['{"action":"final","answer":"ok"}']).run(self.manifest(expected_final="ok"))
        self.assertEqual(rec.status, "PASS")
        self.assertEqual(rec.model_calls, 1)
        self.assertEqual(rec.total_tokens, 15)

    def test_02_tool_then_final(self):
        rec = self.agent([
            '{"action":"tool","tool":"lookup","arguments":{"value":"alpha"}}',
            '{"action":"final","answer":"verified"}',
        ]).run(self.manifest(expected_final="verified"))
        self.assertEqual(rec.status, "PASS")
        self.assertEqual(rec.tool_calls, 1)
        self.assertEqual(rec.model_calls, 2)

    def test_03_exact_tool_evidence(self):
        rec = self.agent([
            '{"action":"tool","tool":"lookup","arguments":{"value":"alpha"}}',
            '{"action":"final","answer":"verified"}',
        ]).run(self.manifest())
        req = [e for e in rec.events if e.event_type == "TOOL_REQUEST"][0]
        self.assertEqual(req.payload["arguments"]["value"], "alpha")

    def test_04_undeclared_tool_fail_closed(self):
        rec = self.agent(['{"action":"tool","tool":"secret","arguments":{}}']).run(self.manifest())
        self.assertEqual(rec.status, "FAIL_CLOSED")
        self.assertEqual(rec.failure_class, "UNDECLARED_TOOL")

    def test_05_malformed_model_json_fail_closed(self):
        rec = self.agent(["not-json"]).run(self.manifest())
        self.assertEqual(rec.failure_class, "MALFORMED_MODEL_RESPONSE")

    def test_06_empty_model_content_fail_closed(self):
        rec = self.agent([None]).run(self.manifest())
        self.assertEqual(rec.failure_class, "MALFORMED_MODEL_RESPONSE")

    def test_07_malformed_tool_fail_closed(self):
        rec = self.agent([
            '{"action":"tool","tool":"lookup","arguments":{}}'
        ], CounterTool(malformed=True)).run(self.manifest())
        self.assertEqual(rec.failure_class, "MALFORMED_TOOL_RESPONSE")

    def test_08_retry_once_then_pass(self):
        tool = CounterTool(failures=1)
        rec = self.agent([
            '{"action":"tool","tool":"lookup","arguments":{"value":"a"}}',
            '{"action":"final","answer":"ok"}',
        ], tool).run(self.manifest())
        self.assertEqual(rec.status, "PASS")
        self.assertEqual(rec.retries_used, 1)
        self.assertEqual(tool.calls, 2)

    def test_09_retry_exhaustion_fail_closed(self):
        tool = CounterTool(failures=2)
        rec = self.agent([
            '{"action":"tool","tool":"lookup","arguments":{"value":"a"}}'
        ], tool).run(self.manifest())
        self.assertEqual(rec.failure_class, "TOOL_TIMEOUT_RETRIES_EXHAUSTED")

    def test_10_step_limit_fail_closed(self):
        outputs = [
            '{"action":"tool","tool":"lookup","arguments":{"value":"a"}}',
            '{"action":"tool","tool":"lookup","arguments":{"value":"b"}}',
        ]
        rec = self.agent(outputs).run(self.manifest(max_steps=2))
        self.assertEqual(rec.failure_class, "STEP_LIMIT_EXCEEDED")

    def test_11_final_assertion_mismatch(self):
        rec = self.agent(['{"action":"final","answer":"actual"}']).run(self.manifest(expected_final="expected"))
        self.assertEqual(rec.status, "FAIL")
        self.assertEqual(rec.failure_class, "FINAL_ASSERTION_MISMATCH")

    def test_12_report_traceable(self):
        rec = self.agent(['{"action":"final","answer":"ok"}']).run(self.manifest())
        report = build_report(rec)
        self.assertEqual(len(report["run_digest"]), 64)
        self.assertGreater(report["event_count"], 0)

    def test_13_nebius_adapter_parses_provider_evidence(self):
        transport = FixtureTransport({
            "id": "abc",
            "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
            "choices": [{"finish_reason": "stop", "message": {"content": "{\"action\":\"final\",\"answer\":\"ok\"}", "reasoning": "brief"}}],
            "usage": {"prompt_tokens": 26, "completion_tokens": 8, "total_tokens": 34},
        })
        adapter = NebiusNemotronAdapter(model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", api_key="secret", transport=transport)
        response = adapter.complete(ModelRequest(messages=({"role":"user","content":"x"},), max_tokens=32))
        self.assertEqual(response.response_id, "abc")
        self.assertEqual(response.usage.total_tokens, 34)
        self.assertEqual(response.finish_reason, "stop")

    def test_14_nebius_secret_not_in_model_response(self):
        transport = FixtureTransport({
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {},
        })
        adapter = NebiusNemotronAdapter(model="m", api_key="TOPSECRET", transport=transport)
        response = adapter.complete(ModelRequest(messages=({"role":"user","content":"x"},)))
        self.assertNotIn("TOPSECRET", repr(response))

    def test_15_nebius_missing_key_fails_before_call(self):
        with self.assertRaises(ProviderError):
            NebiusNemotronAdapter(model="m", api_key="")

    def test_16_nebius_authorization_header_is_runtime_only(self):
        transport = FixtureTransport({
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {},
        })
        adapter = NebiusNemotronAdapter(model="m", api_key="abc123", transport=transport)
        adapter.complete(ModelRequest(messages=({"role":"user","content":"x"},)))
        self.assertEqual(transport.calls[0][1]["Authorization"], "Bearer abc123")

    def test_17_usage_accumulates_across_model_calls(self):
        rec = self.agent([
            '{"action":"tool","tool":"lookup","arguments":{"value":"a"}}',
            '{"action":"final","answer":"ok"}',
        ]).run(self.manifest())
        self.assertEqual(rec.prompt_tokens, 20)
        self.assertEqual(rec.completion_tokens, 10)
        self.assertEqual(rec.total_tokens, 30)

    def test_18_evidence_omits_secret_material(self):
        rec = self.agent(['{"action":"final","answer":"ok"}']).run(self.manifest())
        blob = json.dumps([e.payload for e in rec.events])
        self.assertNotIn("Authorization", blob)
        self.assertNotIn("api_key", blob)


if __name__ == "__main__":
    unittest.main(verbosity=2)
