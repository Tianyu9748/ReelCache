import copy
from pathlib import Path
import tempfile
import unittest

from reelcache.errors import ConfigurationError
from reelcache.synthetic_workloads import HISTORY_LENGTHS, compile_session, read_sessions, validate_bundle, validate_session
from reelcache.workloads import replay_cpu_control, validate_workload


def session(index=0):
    """Explicitly templated test fixture, never a generated benchmark session."""
    return {
        "session_id": f"s{index:04d}", "title": f"Authored unit fixture {index}", "category": "test fixture",
        "prompts": [
            f"A copper toy bird in a miniature paper garden unfolds its wings beside the violet bench for action {p} session {index}."
            for p in range(20)
        ],
        "forks": [
            {"history_prompt_count": h, "change_type": "action",
             "prompts": [
                 f"The copper toy bird in the miniature paper garden inspects a fallen feather beside the violet bench at branch {h} step {p} session {index}."
                 for p in range(2)
             ]}
            for h in HISTORY_LENGTHS
        ],
    }


class SyntheticSessionTests(unittest.TestCase):
    def test_at_least_twenty_main_prompts_and_five_history_lengths(self):
        data = session()
        validate_session(data)
        data["prompts"].append("A copper toy bird folds both wings in the miniature paper garden as evening lanterns illuminate the violet bench.")
        validate_session(data)
        for change in ("short", "missing_history", "bad_type", "future_history", "duplicate_branch"):
            bad = session()
            if change == "short":
                bad["prompts"].pop()
            elif change == "missing_history":
                bad["forks"].pop()
            elif change == "bad_type":
                bad["forks"][0]["change_type"] = "random"
            elif change == "future_history":
                bad["forks"][-1]["history_prompt_count"] = 21
            else:
                bad["forks"][0]["prompts"][1] = bad["forks"][0]["prompts"][0]
            with self.subTest(change=change), self.assertRaises(ConfigurationError):
                validate_session(bad)

    def test_fork_text_must_be_additional_not_copied_main_text(self):
        bad = session()
        bad["forks"][0]["prompts"][0] = bad["prompts"][4]
        with self.assertRaises(ConfigurationError):
            validate_session(bad)

    def test_every_fork_prompt_is_checked_for_self_contained_length(self):
        bad = session()
        bad["forks"][1]["prompts"][1] = "The blue paper bird clears a channel with its brass wheel."
        with self.assertRaisesRegex(ConfigurationError, "18"):
            validate_session(bad)

    def test_bundle_checks_size_order_prefix_and_unique_new_text(self):
        rows = [session(i) for i in range(3)]
        originals = [{"prompts": rows[0]["prompts"][:6]}]
        report = validate_bundle(rows, originals, expected_sessions=3)
        self.assertEqual(report["main_prompts"], 60)
        self.assertEqual(report["fork_prompts"], 30)
        self.assertEqual(report["forks"], 15)
        self.assertEqual(report["preserved_source_sessions"], 1)
        for change in ("count", "order", "prefix", "repeat", "title", "copy_source"):
            bad = copy.deepcopy(rows)
            if change == "count":
                bad.pop()
            elif change == "order":
                bad.reverse()
            elif change == "prefix":
                bad[0]["prompts"][0] = bad[0]["prompts"][9]
            elif change == "repeat":
                bad[2]["prompts"][5] = bad[1]["prompts"][5]
            elif change == "copy_source":
                bad[2]["prompts"][5] = originals[0]["prompts"][0]
            else:
                bad[2]["title"] = bad[1]["title"]
            with self.subTest(change=change), self.assertRaises(ConfigurationError):
                validate_bundle(bad, originals, expected_sessions=3)

    def test_compiler_forks_at_saved_history_and_resumes_unchanged_parent(self):
        workload = compile_session(session())
        validate_workload(workload)
        self.assertEqual(workload["summary"]["operations"], {"generate": 30, "fork": 5, "resume": 5})
        self.assertEqual(len(workload["prompts"]), 30)
        self.assertEqual(workload["summary"]["branches"], 6)
        self.assertEqual(workload["source_provenance"], "llm_authored_synthetic")
        self.assertFalse(workload["claims"]["recorded_agent_trace"])
        forks = [e for e in workload["events"] if e["operation"] == "fork"]
        self.assertEqual([e["history_prompt_count"] for e in forks], list(HISTORY_LENGTHS))
        self.assertEqual([e["target_clean_chunk"] for e in forks], list(HISTORY_LENGTHS))
        for event in forks:
            self.assertEqual(event["parent_branch_id"], "main")
            self.assertEqual(event["provenance_class"], "controlled_synthetic")
            self.assertNotEqual(event["seed"], 0)
        self.assertEqual(compile_session(session()), workload)

    def test_native_chunk_mapping_is_explicit_and_scales_fork_positions(self):
        workload = compile_session(session(), chunks_per_prompt=3, seed=15)
        self.assertEqual(workload["summary"]["operations"]["generate"], 90)
        forks = [e for e in workload["events"] if e["operation"] == "fork"]
        self.assertEqual([e["target_clean_chunk"] for e in forks], [3 * h for h in HISTORY_LENGTHS])
        self.assertFalse(workload["history_mapping"]["native_geometry_qualified"])
        for value in (0, True, 1.5):
            with self.assertRaises(ConfigurationError):
                compile_session(session(), chunks_per_prompt=value)

    def test_all_five_forks_replay_in_real_raw_store(self):
        workload = compile_session(session())
        with tempfile.TemporaryDirectory() as directory:
            report = replay_cpu_control(workload, Path(directory) / "store")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["operations"], workload["summary"]["operations"])
        self.assertFalse(report["video_quality_measured"])
        self.assertEqual(report["source_provenance"], "llm_authored_synthetic")

    def test_jsonl_reader_rejects_malformed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text("{bad json}\n")
            with self.assertRaises(ConfigurationError):
                read_sessions(path)


if __name__ == "__main__":
    unittest.main()
