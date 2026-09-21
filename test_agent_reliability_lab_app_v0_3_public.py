import os
import unittest
from unittest.mock import patch

import agent_reliability_lab_app_v0_3_public as app
from agent_reliability_lab_v0_2 import ModelRequest, ModelResponse, TokenUsage


class ScriptedModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0
    def complete(self, request: ModelRequest):
        self.calls += 1
        return ModelResponse(
            provider="synthetic", model="fixture", response_id=str(self.calls),
            content=self.outputs.pop(0), reasoning=None, finish_reason="stop",
            usage=TokenUsage(3, 2, 5), latency_ms=1,
        )


class TestPublicApp(unittest.TestCase):
    def test_00_judge_ui_features_preserved(self):
        self.assertIn("Reliability result", app.INDEX_HTML)
        self.assertIn("Load judge demo", app.INDEX_HTML)
        self.assertIn("What the agent actually did", app.INDEX_HTML)
        self.assertIn("Show raw structured evidence", app.INDEX_HTML)

    def test_01_public_ui_excludes_tavily_and_model_override(self):
        self.assertNotIn("Tavily", app.INDEX_HTML)
        self.assertNotIn('id="mode"', app.INDEX_HTML)
        self.assertNotIn('id="model"', app.INDEX_HTML)
        self.assertIn("Local supplied evidence only", app.INDEX_HTML)

    def test_02_evidence_search(self):
        tool = app.build_evidence_search("Launch October 12\nBudget approved\nLaunch delayed to October 19")
        out = tool({"query": "launch October 19"})
        self.assertEqual(out["value"][0]["evidence_id"], "E3")

    def test_03_local_tool_chain(self):
        model = ScriptedModel([
            '{"action":"tool","tool":"evidence_search","arguments":{"query":"release approval"}}',
            '{"action":"final","answer":"Public release is not authorized."}',
        ])
        out = app.run_demo({"goal":"verify release", "evidence":"No public-release approval has been issued."}, model_adapter=model)
        self.assertEqual(out["report"]["status"], "PASS")
        self.assertEqual(out["report"]["tool_calls"], 1)
        reqs = [e for e in out["events"] if e["event_type"] == "TOOL_REQUEST"]
        self.assertEqual([e["payload"]["name"] for e in reqs], ["evidence_search"])

    def test_04_no_evidence_direct(self):
        model = ScriptedModel(['{"action":"final","answer":"uncertain"}'])
        out = app.run_demo({"goal":"answer carefully"}, model_adapter=model)
        self.assertEqual(out["report"]["final_output"], "uncertain")

    def test_05_goal_required(self):
        with self.assertRaises(ValueError):
            app.run_demo({"goal":""}, model_adapter=ScriptedModel([]))

    def test_06_size_limits(self):
        with self.assertRaises(ValueError):
            app.run_demo({"goal":"x"*5000}, model_adapter=ScriptedModel([]))
        with self.assertRaises(ValueError):
            app.run_demo({"goal":"x", "evidence":"y"*(21*1024)}, model_adapter=ScriptedModel([]))

    def test_07_browser_payload_omits_secrets(self):
        model = ScriptedModel(['{"action":"final","answer":"ok"}'])
        out = app.run_demo({"goal":"x"}, model_adapter=model)
        text = str(out)
        self.assertNotIn("Authorization", text)
        self.assertNotIn("api_key", text)

    def test_08_rate_limit_three_per_hour(self):
        with app._RATE_LOCK:
            app._IP_RUNS.clear(); app._DAILY_STATE["day"] = None; app._DAILY_STATE["accepted"] = 0
        now = 100000.0
        self.assertTrue(app._reserve_run("1.2.3.4", now)[0])
        self.assertTrue(app._reserve_run("1.2.3.4", now+1)[0])
        self.assertTrue(app._reserve_run("1.2.3.4", now+2)[0])
        ok, why = app._reserve_run("1.2.3.4", now+3)
        self.assertFalse(ok); self.assertEqual(why, "rate limit reached")

    def test_09_daily_limit(self):
        old = app.DAILY_RUN_LIMIT
        try:
            app.DAILY_RUN_LIMIT = 2
            with app._RATE_LOCK:
                app._IP_RUNS.clear(); app._DAILY_STATE["day"] = None; app._DAILY_STATE["accepted"] = 0
            self.assertTrue(app._reserve_run("a", 200000.0)[0])
            self.assertTrue(app._reserve_run("b", 200001.0)[0])
            ok, why = app._reserve_run("c", 200002.0)
            self.assertFalse(ok); self.assertEqual(why, "demo budget reached")
        finally:
            app.DAILY_RUN_LIMIT = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
