import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from reelcache.errors import ConfigurationError
from reelcache.workloads import (
    WorkloadPolicy, browser_observations, compile_browser_workload, read_narratives,
    replay_cpu_control, validate_workload,
)


def observed(index, intent, tab="0", stamp=None, *, document=None, origin=None, back=False, url=None):
    return {
        "source_event_index": index, "source_timestamp": index if stamp is None else stamp,
        "source_type": "browser", "source_action": intent, "tab_id": tab,
        "origin_tab_id": origin, "source_state_id": document, "back_forward": back,
        "source_url_id": url,
    }


NARRATIVE = {"narrative_id": "fixture", "prompts": ["A robot stands.", "The robot walks.", "The robot waves."]}


def compile_(rows, **options):
    return compile_browser_workload(rows, NARRATIVE, WorkloadPolicy(chunk_seconds=1, **options),
                                    source_id="authored-test-fixture", episode_id="episode",
                                    source_provenance="authored_fixture")


class NarrativeAndObservationTests(unittest.TestCase):
    def test_narratives_are_group_identified_and_duplicates_are_rejected(self):
        rows = read_narratives('{"prompts":["a","b"]}\n\n{"prompts":["c","d"]}\n')
        self.assertNotEqual(rows[0]["narrative_id"], rows[1]["narrative_id"])
        self.assertEqual(rows, read_narratives('{"prompts":["a","b"]}\n{"prompts":["c","d"]}'))
        for text in ("", "{", '{"prompts":[]}', '{"prompts":[""]}', '{"prompts":["a"]}\n{"prompts":["a"]}'):
            with self.subTest(text=text), self.assertRaises(ConfigurationError):
                read_narratives(text)

    def test_raw_parser_keeps_event_order_timestamps_tabs_but_not_dialogue(self):
        replay = {"data": [
            {"type": "chat", "timestamp": -1, "utterance": "private test dialogue"},
            {"type": "browser", "timestamp": 2, "action": {"intent": "load", "arguments": {
                "metadata": {"tabId": 8, "url": "https://example.invalid/private"},
                "properties": {"documentId": "D"},
            }}},
            {"type": "browser", "timestamp": 3, "action": {"intent": "tabswitch", "arguments": {
                "metadata": {"tabId": -1}, "properties": {"tabId": 9, "tabIdOrigin": 8},
            }}},
        ]}
        result = browser_observations(replay)
        self.assertEqual([r["source_event_index"] for r in result], [0, 1, 2])
        self.assertEqual(result[2]["tab_id"], "9")
        self.assertEqual(result[2]["origin_tab_id"], "8")
        self.assertNotIn("private", json.dumps(result))
        self.assertEqual(result[1]["source_timestamp"], 2)

    def test_invalid_policies_and_trace_rows_fail_explicitly(self):
        for kwargs in ({"chunk_seconds": 0}, {"chunk_seconds": float("nan")}, {"seed": -1},
                       {"pause_gap_seconds": -1}, {"chunks_per_action": True}, {"branch_content": "random"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ConfigurationError):
                WorkloadPolicy(**{"chunk_seconds": 1, **kwargs})
        for replay in ({}, {"data": [3]}, {"data": [{"timestamp": float("inf")}]},
                       {"data": [{"action": {"arguments": {"properties": []}}}]}):
            with self.assertRaises(ConfigurationError):
                browser_observations(replay)


class TraceMappingTests(unittest.TestCase):
    def test_fork_resume_and_pause_follow_source_events(self):
        rows = [
            observed(0, "load", stamp=0, document="A"),
            observed(1, "tabcreate", tab="1", stamp=1, origin="0"),
            observed(2, "tabswitch", tab="1", stamp=2, origin="0"),
            observed(3, "click", tab="1", stamp=3),
            observed(4, "tabswitch", tab="0", stamp=40, origin="1"),
            observed(5, "click", tab="0", stamp=41),
        ]
        workload = compile_(rows)
        ops = workload["summary"]["operations"]
        self.assertEqual(ops, {"generate": 3, "fork": 1, "resume": 3, "pause": 1})
        fork = next(e for e in workload["events"] if e["operation"] == "fork")
        self.assertEqual(fork["source_event_index"], 1)
        self.assertEqual(fork["parent_branch_id"], "main")
        self.assertEqual(fork["target_checkpoint"], workload["events"][0]["output_checkpoint"])
        self.assertEqual(fork["provenance_class"], "trace_mapped")
        self.assertFalse(workload["claims"]["recorded_video_user_trace"])
        self.assertEqual([r["source_event_index"] for r in workload["source_audit"]], list(range(6)))
        validate_workload(workload)

    def test_branch_seed_is_deterministic_and_next_prompt_rule_is_explicit(self):
        rows = [observed(0, "load"), observed(1, "tabcreate", tab="1", origin="0")]
        a = compile_(rows)
        self.assertEqual(a, compile_(rows))
        new_seed = next(e for e in a["events"] if e["operation"] == "fork")
        self.assertNotEqual(new_seed["seed"], 0)
        b = compile_(rows, branch_content="next_prompt")
        changed = next(e for e in b["events"] if e["operation"] == "fork")
        self.assertEqual(changed["prompt_index"], 1)
        self.assertEqual(changed["content_policy"], "next_prompt")

    def test_missing_opener_is_labeled_inferred_and_unknown_tabs_are_audited(self):
        rows = [observed(0, "load"), observed(1, "tabcreate", tab="1"),
                observed(2, "tabswitch", tab="unknown")]
        workload = compile_(rows)
        self.assertEqual(workload["summary"]["operations"]["fork"], 1)
        self.assertEqual(next(e for e in workload["events"] if e["operation"] == "fork")["provenance_class"],
                         "inferred_relationship")
        self.assertEqual(workload["summary"]["unmapped_reasons"], {"target_tab_has_no_mapped_branch": 1})

    def test_rewind_uses_immutable_checkpoint_and_old_suffix_can_be_resumed(self):
        workload = compile_([
            observed(0, "load", document="A"), observed(1, "load", document="B"),
            observed(2, "load", document="A", back=True),
            observed(3, "load", document="B", back=True),
        ])
        generations = [e for e in workload["events"] if e["operation"] == "generate"]
        self.assertEqual(len(set(e["output_checkpoint"] for e in generations)), 4)
        rewind = next(e for e in workload["events"] if e["operation"] == "rewind")
        self.assertEqual(rewind["target_checkpoint"], generations[0]["output_checkpoint"])
        resume = next(e for e in workload["events"] if e["operation"] == "resume")
        self.assertEqual(resume["target_checkpoint"], generations[1]["output_checkpoint"])
        validate_workload(workload)

    def test_url_return_is_not_an_exact_state_rewind(self):
        rows = [observed(0, "load", url="u1"), observed(1, "load", url="u2"),
                observed(2, "load", url="u1", back=True)]
        workload = compile_(rows)
        self.assertNotIn("rewind", workload["summary"]["operations"])
        inferred = compile_(rows, infer_url_rewind=True)
        rewind = next(e for e in inferred["events"] if e["operation"] == "rewind")
        self.assertEqual(rewind["provenance_class"], "inferred_relationship")

    def test_no_lifecycle_operations_are_invented_for_a_linear_trace(self):
        workload = compile_([observed(i, "click") for i in range(4)])
        self.assertEqual(workload["summary"]["operations"], {"generate": 4})
        self.assertEqual(workload["summary"]["branches"], 1)

    def test_source_clock_and_event_order_are_not_silently_repaired(self):
        for rows in ([observed(1, "click"), observed(0, "click")],
                     [observed(0, "click", stamp=2), observed(1, "click", stamp=1)]):
            with self.assertRaises(ConfigurationError):
                compile_(rows)
        negative = compile_([observed(0, "click", stamp=-2), observed(1, "click", stamp=1)])
        self.assertEqual(negative["source_audit"][0]["mapping_status"], "pre_recording")
        self.assertEqual(negative["summary"]["operations"], {"generate": 1})

    def test_validator_rejects_dangling_parent_and_checkpoint_collisions(self):
        workload = compile_([observed(0, "load"), observed(1, "tabcreate", tab="1", origin="0")])
        bad = copy.deepcopy(workload)
        bad["events"][-1]["target_checkpoint"] = "missing"
        with self.assertRaises(ConfigurationError):
            validate_workload(bad)
        bad = compile_([observed(0, "load"), observed(1, "click")])
        bad["events"][-1]["output_checkpoint"] = bad["events"][0]["output_checkpoint"]
        with self.assertRaises(ConfigurationError):
            validate_workload(bad)

    def test_cli_license_gate_runs_before_input_reads(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([
            sys.executable, "-B", str(root / "examples/interactive_workloads.py"), "compile",
            "--replay", "missing.json", "--narratives", "missing.jsonl",
            "--source-id", "test", "--episode-id", "test", "--chunk-seconds", "1", "--output", "missing-output.json",
        ], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("research-use-cleared", result.stderr)
        self.assertNotIn("FileNotFoundError", result.stderr)

    def test_cli_validates_an_authored_fixture_without_loading_a_model(self):
        root = Path(__file__).resolve().parents[1]
        workload = compile_([observed(0, "load"), observed(1, "click")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.json"
            path.write_text(json.dumps(workload))
            result = subprocess.run([
                sys.executable, "-B", str(root / "examples/interactive_workloads.py"), "validate", str(path),
            ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["native_model_executed"])

    def test_real_store_replays_fork_rewind_resume_and_reopen_without_changing_raw_history(self):
        workload = compile_([
            observed(0, "load", document="A"),
            observed(1, "click", document="B"),
            observed(2, "tabcreate", tab="1", origin="0"),
            observed(3, "tabswitch", tab="1", origin="0"),
            observed(4, "click", tab="1"),
            observed(5, "tabswitch", tab="0", origin="1"),
            observed(6, "load", document="A", stamp=45),
            observed(7, "load", document="B", stamp=46),
            observed(8, "tabremove", tab="1", stamp=47),
        ])
        with tempfile.TemporaryDirectory() as directory:
            report = replay_cpu_control(workload, Path(directory) / "store")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["operations"]["fork"], 1)
        self.assertEqual(report["operations"]["rewind"], 1)
        self.assertEqual(report["operations"]["pause"], 1)
        self.assertEqual(report["operations"]["retire"], 1)
        self.assertFalse(report["video_quality_measured"])
        self.assertEqual(report["source_provenance"], "authored_fixture")


if __name__ == "__main__":
    unittest.main()
