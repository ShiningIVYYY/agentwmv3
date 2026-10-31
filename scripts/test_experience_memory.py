#!/usr/bin/env python3
"""CPU unit fixtures only. Synthetic fixtures are NOT benchmark results."""
import copy
import json
from pathlib import Path
import sys
import time
import unittest
import tempfile
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experience.schema import ExternalQuery, Truth, validate_case
from experience.collection import receipts
from experience.splits import digest, goal_key, validate_manifest
from experience.synthesis import applicable, synthesize, validate_card
from experience.memory import ExperienceMemory, bm25
from scripts.build_experience_memory import build

OBS = "This room is called the bathroom. You also see:\n\tA door to the kitchen (that is closed)"


def fixture():
    rows, cases = [], []
    for i in (1, 2):
        goal = "Unit fixture task " + str(i)
        row = {"task_id": "fixture_" + str(i), "taskName": "fixture", "variationIdx": i,
               "goal_sha256": digest(goal), "goal_group": goal_key(goal)}
        rows.append(row)
        cases.append({"schema": "experience-case-v1", "case_id": "case" + str(i),
            "benchmark": "scienceworld", "task_id": row["task_id"], "canonical_instance": ["fixture", i],
            "goal_group": row["goal_group"], "episode_id": "episode" + str(i), "split": "train",
            "goal": goal, "subgoal": None, "observation": OBS, "premise": "The kitchen is accessible.",
            "expected_consequence": "I will enter the kitchen.", "draft_action": "go kitchen",
            "executed_action": "go kitchen", "environment_feedback": "The door is not open.",
            "reward": 0, "task_score": 0, "done": False, "success": None, "wm_called": False,
            "source_trajectory_id": "trajectory" + str(i), "step_index": 1,
            "decision_at_ns": 100, "outcome_available_at_ns": 200,
            "provenance": {"oracle_used": False}, "transition_fingerprint": "unit_fixture"})
    manifest = {"partitions": {"train": rows, "development": [], "test": []}}
    return manifest, cases


def query(**kwargs):
    return ExternalQuery(**dict({"benchmark": "scienceworld", "task_id": "dev_fixture", "episode_id": "dev_episode",
        "step_index": 1, "available_at_ns": time.time_ns() + 1000000, "goal": "Go to the kitchen",
        "observation": OBS, "draft_action": "go kitchen", "premise": "The kitchen is accessible.",
        "expected_consequence": "I will enter the kitchen."}, **kwargs))


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.cases = fixture()

    def memory(self, cases=None):
        return ExperienceMemory(build(cases if cases is not None else self.cases, self.manifest), self.manifest)

    def test_schema(self):
        self.assertEqual(validate_case(self.cases[0], self.manifest), self.cases[0])

    def test_missing_required(self):
        del self.cases[0]["environment_feedback"]
        with self.assertRaises(ValueError): validate_case(self.cases[0], self.manifest)

    def test_extra_oracle(self):
        self.cases[0]["oracle_state"] = {}
        with self.assertRaises(ValueError): validate_case(self.cases[0], self.manifest)

    def test_query_rejects_future(self):
        for key in ("environment_feedback", "reward", "next_observation", "success", "oracle_state"):
            with self.assertRaises(TypeError): query(**{key: "future"})

    def test_no_implicit_boolean(self):
        for value in Truth:
            with self.assertRaises(TypeError): bool(value)

    def test_dev_case_rejected(self):
        self.cases[0]["split"] = "development"
        with self.assertRaises(ValueError): synthesize(self.cases, self.manifest)

    def test_id_leak(self):
        self.manifest["partitions"]["test"] = [dict(self.manifest["partitions"]["train"][0])]
        with self.assertRaises(ValueError): validate_manifest(self.manifest)

    def test_goal_leak(self):
        self.manifest["partitions"]["development"] = [dict(self.manifest["partitions"]["train"][0], task_id="different")]
        with self.assertRaises(ValueError): validate_manifest(self.manifest)

    def test_case_goal_changed(self):
        self.cases[0]["goal"] = "invented"
        with self.assertRaises(ValueError): validate_case(self.cases[0], self.manifest)

    def test_bootstrap_not_valid_for_construction(self):
        self.manifest["partitions"]["test"] = [{"task_id": "other_test", "goal_sha256": "not_train",
                                                "goal_group": None}]
        with self.assertRaises(ValueError): self.memory()

    def test_temporal_case(self):
        self.cases[0]["decision_at_ns"] = 300
        with self.assertRaises(ValueError): validate_case(self.cases[0], self.manifest)

    def test_missing_success_unknown(self):
        self.assertIsNone(validate_case(self.cases[0], self.manifest)["success"])

    def test_nonfinite_outcome(self):
        self.cases[0]["task_score"] = float("nan")
        with self.assertRaises(ValueError): validate_case(self.cases[0], self.manifest)

    def test_single_case_is_limited(self):
        result = self.memory(self.cases[:1]).evaluate(query())
        self.assertEqual([result[k] for k in ("coverage", "conflict", "sufficiency")], ["unknown"] * 3)

    def test_independent_instances(self):
        self.cases[1]["episode_id"] = self.cases[0]["episode_id"]
        cards, _ = synthesize(self.cases, self.manifest)
        self.assertEqual(cards[0]["reliability"], "limited")

    def test_duplicate_case(self):
        with self.assertRaises(ValueError): synthesize([self.cases[0], self.cases[0]], self.manifest)

    def test_clone_source(self):
        self.cases[1]["source_trajectory_id"] = self.cases[0]["source_trajectory_id"]
        with self.assertRaises(ValueError): synthesize(self.cases, self.manifest)

    def test_supported_conflict(self):
        r = self.memory().evaluate(query())
        self.assertEqual((r["coverage"], r["conflict"], r["sufficiency"]), ("true", "true", "false"))
        self.assertEqual(set(r["supporting_evidence_ids"]), {"case1", "case2"})
        self.assertFalse(r["action_modified"])

    def test_counterexample_abstains(self):
        self.cases[1]["environment_feedback"] = "You move to the kitchen."
        r = self.memory().evaluate(query())
        self.assertEqual(r["conflict"], "unknown")
        self.assertEqual(r["sufficiency"], "unknown")
        self.assertEqual(r["supporting_evidence_ids"], [])

    def test_applicability_not_similarity(self):
        r = self.memory().evaluate(query(draft_action="look around"))
        self.assertTrue(r["retrieved_cards"])
        self.assertEqual(r["matched_experience_ids"], [])
        self.assertEqual(r["coverage"], "unknown")

    def test_empty_memory(self):
        r = self.memory([]).evaluate(query())
        self.assertEqual(r["retrieved_cards"], [])
        self.assertEqual(r["conflict"], "unknown")

    def test_future_snapshot(self):
        r = self.memory().evaluate(query(available_at_ns=300))
        self.assertEqual(r["retrieved_cards"], [])

    def test_current_task_excluded(self):
        r = self.memory().evaluate(query(task_id="fixture_1"))
        self.assertEqual(r["coverage"], "unknown")

    def test_current_episode_excluded(self):
        r = self.memory().evaluate(query(episode_id="episode1"))
        self.assertEqual(r["coverage"], "unknown")

    def test_invented_support(self):
        snap = build(self.cases, self.manifest)
        snap["cards"][0]["supporting_cases"].append("invented")
        with self.assertRaises(KeyError): ExperienceMemory(snap, self.manifest)

    def test_inflated_reliability(self):
        snap = build(self.cases, self.manifest)
        snap["cards"][0]["independent_episodes"] = 100
        with self.assertRaises(ValueError): ExperienceMemory(snap, self.manifest)

    def test_split_identity(self):
        snap = build(self.cases, self.manifest)
        snap["manifest_sha256"] = "other"
        with self.assertRaises(ValueError): ExperienceMemory(snap, self.manifest)

    def test_no_wm_causal_claim(self):
        snap = build(self.cases, self.manifest)
        snap["cards"][0]["wm_utility"] = "positive"
        with self.assertRaises(ValueError): ExperienceMemory(snap, self.manifest)

    def test_bm25_empty_and_topk(self):
        cards, _ = synthesize(self.cases, self.manifest)
        self.assertEqual(bm25(cards, "xyzzy", 1), [])
        with self.assertRaises(ValueError): bm25(cards, "door", 0)

    def test_deterministic_ranking(self):
        cards, _ = synthesize(self.cases, self.manifest)
        self.assertEqual(bm25(cards, "door", 1), bm25(list(reversed(cards)), "door", 1))

    def test_open_sufficiency_scope(self):
        for c in self.cases:
            c["draft_action"] = c["executed_action"] = "open door to the kitchen"
            c["environment_feedback"] = "The door is now open."
        memory = self.memory()
        r = memory.evaluate(query(draft_action="open door to the kitchen", premise="The door is closed.", expected_consequence="The door will open."))
        self.assertEqual((r["conflict"], r["sufficiency"]), ("false", "true"))
        r = memory.evaluate(query(draft_action="open door to the kitchen", expected_consequence="I will complete the task."))
        self.assertEqual(r["sufficiency"], "unknown")

    def test_costs(self):
        r = self.memory().evaluate(query())
        self.assertGreaterEqual(r["retrieval_latency"], 0)
        self.assertGreaterEqual(r["judge_latency"], 0)
        self.assertEqual((r["additional_tokens"], r["judge_model_calls"], r["extra_forward_passes"]), (0, 0, 0))

    def test_unit_wording_alias_only(self):
        self.assertEqual(goal_key("above 0 degrees celsius"), goal_key("above 0 degrees"))
        self.assertNotEqual(goal_key("above 0 degrees"), goal_key("above 100 degrees"))

    def test_normalized_test_goal_leak(self):
        self.manifest["partitions"]["test"] = [{"task_id": "heldout", "goal_sha256": "different_raw_hash",
            "goal_group": self.manifest["partitions"]["train"][0]["goal_group"]}]
        with self.assertRaises(ValueError): validate_manifest(self.manifest)

    def test_sufficiency_abstains_on_unverified_premise(self):
        for c in self.cases:
            c["draft_action"] = c["executed_action"] = "open door to the kitchen"
            c["environment_feedback"] = "The door is now open."
        r = self.memory().evaluate(query(draft_action="open door to the kitchen", premise="The object is conductive.",
                                         expected_consequence="The door will open."))
        self.assertEqual(r["sufficiency"], "unknown")

    def test_query_unchanged(self):
        q = query()
        before = repr(q)
        self.memory().evaluate(q)
        self.assertEqual(repr(q), before)

    def test_receipts_survive_downstream_failure(self):
        fake = SimpleNamespace(http=lambda *args, **kwargs: {"observation": "unit fixture"})
        original = fake.http
        with tempfile.TemporaryDirectory() as directory:
            sink = Path(directory) / "receipts.jsonl"
            with self.assertRaises(RuntimeError):
                with receipts(fake, sink):
                    fake.http("POST", "http://unit-fixture/step", json={"action": "wait"})
                    raise RuntimeError("unit downstream validation failure")
            self.assertEqual(json.loads(sink.read_text())["response"], {"observation": "unit fixture"})
        self.assertIs(fake.http, original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
