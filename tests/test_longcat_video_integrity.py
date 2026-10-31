"""CPU tests of the LongCat-Video integrity defaults (CR-17-1; no SGLang, no GPU).

The LongCat session provider and its ReelCache backend serve with ``verify_reads`` on, no hash audit of the whole store
at open and ``on_open_error="quarantine"`` (``reelcache_backend.INTEGRITY_DEFAULTS``):

* a corrupt archived packet fails the request that would restore it with a typed ``IntegrityError`` before any
  output; the epoch is quarantined, a fork or rewind the request applied is rolled back, the source session is
  unchanged and the failed restore's buffers are released;
* damage found when the store is opened (a quarantined cache lineage or RAW branch, a damaged session record,
  boundary chain or named checkpoint) does not stop the attach: the damaged session is listed as quarantined, ops on
  it fail with ``integrity`` (``describe`` reports it, ``delete`` removes it), epochs a damaged archive may hold fail
  typed when needed, every other session works and garbage collection keeps what it cannot account for;
* ``on_open_error="raise"`` and ``verify_reads=False`` keep the old behaviors (the latter restores a corrupt RAW
  packet silently -- why verification is the default).

The sessions are driven like the port's stages drive them (``tests.test_longcat_video_session_provider``).
"""

from __future__ import annotations

import gc
import os
import shutil
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

import torch

from reelcache_sglang_models.longcat_video import conditioning_cache as cc
from reelcache_sglang_models.longcat_video import reelcache_backend as mb
from reelcache_sglang_models.longcat_video import session_provider as sp
from tests.test_longcat_video_session_provider import (
    ARCH,
    FINGERPRINT,
    conditioning_cache,
    initial_frames,
    make_provider,
    run_request,
)


def corrupt(store, object_id: str) -> None:
    """Flip bytes in the middle of a stored object (same length: only a hash check notices)."""
    path = store.path_for(object_id)
    data = bytearray(path.read_bytes())
    start = len(data) // 2
    for index in range(start, min(start + 64, len(data))):
        data[index] ^= 0x5A
    os.chmod(path, 0o644)
    path.write_bytes(bytes(data))


def blobs(store, root: str) -> list[str]:
    """Every blob a stored object reaches, largest first."""
    seen, stack, out = set(), [root], []
    while stack:
        item = stack.pop()
        if item in seen:
            continue
        seen.add(item)
        if item.startswith("manifest:"):
            stack.extend(store.references(item))
        elif item.startswith("blob:"):
            out.append(item)
    return sorted(out, key=store.object_size, reverse=True)


def segment_epochs(provider, sid: str) -> list[cc.ConditioningCacheEpoch]:
    archive = provider.archive
    return [cc.ConditioningCacheEpoch.from_dict(boundary.segment["epoch"])
            for boundary in archive.path(archive.session(sid)["head"])[1:]]


def packet_blob(provider, epoch: cc.ConditioningCacheEpoch) -> str:
    """The coefficient payload blob of an epoch's node pack (svd_hevc)."""
    runtime = provider.runtime
    node = runtime.load_node(provider.backend.node_ref(epoch))
    records = list(node.matrix_records) or [pair for item in runtime.load_epoch(node.epoch_ref).ranges
                                            for pair in item.records]
    packet = sorted({runtime.load_record(ref).packet_ref for _, ref in records})[0]
    return blobs(runtime.store, packet)[0]


def code_of(exc: BaseException) -> str:
    from reelcache.integrations.sglang.controls import error_of

    return error_of(exc).code


class LongCatIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="longcat-integrity-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def provider(self, name: str = "store", **config) -> sp.LongCatSessionProvider:
        provider = make_provider(self.root / name, **config)
        self.addCleanup(self._close, provider)
        return provider

    @staticmethod
    def _close(provider):
        try:
            provider.close()
        except Exception:  # noqa: BLE001 - already closed by the test
            pass

    def assert_control_error(self, provider, code: str, op: str, **args):
        from reelcache.integrations.sglang.controls import SessionControlError

        with self.assertRaises(SessionControlError) as caught:
            provider.control(op, args)
        self.assertEqual(caught.exception.code, code, caught.exception.to_dict())
        return caught.exception

    # ------------------------------------------------------------------------------------------------ defaults
    def test_serving_defaults_verify_reads_without_an_open_audit(self):
        from reelcache.errors import ConfigurationError
        from reelcache.store import ObjectStore

        self.assertEqual(mb.INTEGRITY_DEFAULTS, {"verify_reads": True, "audit_on_open": False,
                                                 "on_open_error": "quarantine"})
        config = sp.LongCatSessionConfig.from_dict({"store_root": "x"})
        self.assertEqual(config.integrity(), mb.INTEGRITY_DEFAULTS)
        for bad in ({"verify_reads": "yes"}, {"audit_on_open": None}, {"on_open_error": "ignore"}):
            with self.assertRaises(ConfigurationError):
                sp.LongCatSessionConfig.from_dict({"store_root": "x", **bad})
        for bad in ({"verify_reads": 1}, {"audit_on_open": "no"}, {"on_open_error": "skip"}):
            with self.assertRaises(cc.ConditioningCacheError):
                mb.open_runtime(self.root / "bad", mb.model_identity(FINGERPRINT, ARCH), **bad)
        provider = self.provider()
        store, runtime = provider.runtime.store, provider.runtime
        self.assertEqual((store.verify_reads, store.verify_references, runtime.audit_on_open, runtime.on_open_error),
                         (True, False, False, "quarantine"))
        caps = provider.control("capabilities", {})["integrity"]
        self.assertEqual({key: caps[key] for key in ("verify_reads", "verify_references", "audit_on_open",
                                                     "on_open_error")},
                         {"verify_reads": True, "verify_references": False, "audit_on_open": False,
                          "on_open_error": "quarantine"})
        self.assertFalse(any(caps["damage"].values()), caps["damage"])
        self.assertEqual(provider.startup["integrity"]["on_open_error"], "quarantine")
        run_request(provider, {"session_id": "a", "action": "generate", "durable": True}, frames=initial_frames(1),
                    num_segments=2)
        epochs = segment_epochs(provider, "a")
        corrupt(store, packet_blob(provider, epochs[1]))
        provider.close()
        # reopening hashes no stored bytes (the graph check only): the corrupt payload goes unnoticed until read
        with mock.patch.object(ObjectStore, "verify", autospec=True, side_effect=ObjectStore.verify) as audit:
            reopened = self.provider()
        self.assertEqual(audit.call_count, 0)
        self.assertFalse(any(reopened.startup["integrity"]["damage"].values()))
        self.assertEqual(reopened.backend.archive_state(epochs[1], "svd_hevc"), "published")
        reopened.close()
        # an explicit audit at open finds it: the lineage holding it is quarantined, the attach still succeeds
        audited = self.provider(audit_on_open=True)
        damage = audited.startup["integrity"]["damage"]
        self.assertEqual(set(damage["runtime_quarantined"]), {"lineages/" + mb.lineage_branch("a")})
        self.assertEqual(set(damage["lineages"]), {mb.lineage_branch("a")})
        self.assertEqual(audited.backend.archive_state(epochs[1], "svd_hevc"), "quarantined")
        audited.close()
        # the old policy fails the attach on the same damage (typed)
        from reelcache.errors import IntegrityError
        with self.assertRaises(IntegrityError) as caught:
            make_provider(self.root / "store", audit_on_open=True, on_open_error="raise")
        self.assertEqual(code_of(caught.exception), "integrity")

    # ----------------------------------------------------------------------------- corrupt packet (svd_hevc)
    def test_corrupt_packet_fails_typed_without_output_and_changes_no_session(self):
        from reelcache.errors import IntegrityError

        provider = self.provider()
        first = run_request(provider, {"session_id": "a", "action": "generate", "durable": True},
                            frames=initial_frames(2), num_segments=2)
        epochs = segment_epochs(provider, "a")
        self.assertEqual({provider.backend.archive_state(epoch, "svd_hevc") for epoch in epochs}, {"published"})
        before = provider.control("describe", {"session_id": "a", "history": 0})
        corrupt(provider.runtime.store, packet_blob(provider, epochs[1]))
        # fork at boundary 1: segment 1 restores the corrupt epoch -> typed error, no output, the fork is undone
        with self.assertRaises(IntegrityError) as caught:
            run_request(provider, {"session_id": "b", "action": "fork", "from_session": "a", "checkpoint": 1})
        self.assertEqual(code_of(caught.exception), "integrity")
        self.assertIn("is damaged and was not restored: IntegrityError", str(caught.exception))
        summary = list(provider._requests.values())[-1]
        self.assertEqual((summary["error_code"], summary["generated_segments"], summary["segments"]),
                         ("integrity", 1, []))
        self.assertEqual(summary["rolled_back"], {"action": "fork", "removed": "b"})
        self.assertNotIn("b", [entry["session_id"] for entry in provider.control("list", {})["sessions"]])
        self.assertNotIn(mb.lineage_branch("b"), provider.runtime.list_lineages())
        self.assertEqual(provider.backend.archive_state(epochs[1], "svd_hevc"), "quarantined")
        self.assertEqual(provider.backend.archive_state(epochs[0], "svd_hevc"), "published")
        after = provider.control("describe", {"session_id": "a", "history": 0})
        self.assertEqual((after["head"], after["generated_segments"]), (before["head"], before["generated_segments"]))
        self.assertEqual([item["cache_epoch_now"] for item in after["segments"]], ["published", "quarantined"])
        damage = after["damage"]["epochs"]
        self.assertEqual([(item["segment_index"], item["representation"]) for item in damage], [(1, "svd_hevc")])
        # the quarantined epoch fails again without reading; a rewind onto it is undone
        with mock.patch.object(provider.backend, "_restore_lossy") as restore:
            with self.assertRaisesRegex(IntegrityError, "quarantined"):
                run_request(provider, {"session_id": "c", "action": "fork", "from_session": "a", "checkpoint": 1})
        restore.assert_not_called()
        record = provider.runtime.store.get_ref("longcat/sessions/a")
        with self.assertRaises(IntegrityError):
            run_request(provider, {"session_id": "a", "action": "rewind", "checkpoint": 1})
        summary = list(provider._requests.values())[-1]
        self.assertEqual(summary["rolled_back"], {"action": "rewind", "head": before["head"]})
        state = provider.control("describe", {"session_id": "a", "history": 0})
        self.assertEqual((state["head"], state["status"]), (before["head"], "idle"))
        self.assertEqual(provider.runtime.store.get_ref("longcat/sessions/a"), record)  # the same record object
        # the healthy epoch and the rest of the session keep working: fork at 0 restores segment 0, resume builds
        forked = run_request(provider, {"session_id": "d", "action": "fork", "from_session": "a", "checkpoint": 0})
        self.assertEqual(forked["summary"]["segments"][0]["provider_cache"], "restore")
        self.assertTrue((forked["outputs"][0] == first["outputs"][0]).all())
        resumed = run_request(provider, {"session_id": "a", "action": "resume"})
        self.assertEqual(resumed["summary"]["segments"][0]["provider_cache"], "build")
        self.assertIsNone(provider._run)

    def test_failed_restore_releases_its_buffers_while_the_error_lives(self):
        from reelcache.errors import IntegrityError

        provider = self.provider()
        run_request(provider, {"session_id": "a", "action": "generate", "durable": True}, frames=initial_frames(3))
        epoch = segment_epochs(provider, "a")[0]
        corrupt(provider.runtime.store, packet_blob(provider, epoch))
        backend = provider.backend
        allocated, real = [], backend._allocate

        def tracking(epoch, device):
            keys, values = real(epoch, device)
            allocated.extend(weakref.ref(tensor) for tensor in keys + values)
            return keys, values

        backend.arm_restore("svd_hevc")
        try:
            with mock.patch.object(backend, "_allocate", side_effect=tracking):
                with self.assertRaises(IntegrityError) as caught:
                    backend.restore(epoch)
        finally:
            backend.arm_restore(None)
        gc.collect()
        self.assertEqual(len(allocated), 2 * ARCH.depth)
        self.assertTrue(all(ref() is None for ref in allocated))  # freed although the error is still referenced
        self.assertIsNotNone(caught.exception.__cause__)
        self.assertEqual(backend.archive_state(epoch, "svd_hevc"), "quarantined")
        self.assertEqual([event["event"] for event in backend.events if event["event"] == "quarantined"],
                         ["quarantined"])

    def test_store_index_failures_are_not_quarantined_as_damage(self):
        import sqlite3

        from reelcache.errors import IntegrityError

        provider = self.provider()
        run_request(provider, {"session_id": "a", "action": "generate", "durable": True}, frames=initial_frames(5))
        epoch = segment_epochs(provider, "a")[0]
        backend = provider.backend
        try:
            raise IntegrityError("object store index operation failed") from sqlite3.OperationalError("disk I/O error")
        except IntegrityError as exc:
            index_failure = exc
        self.assertFalse(mb.is_damage(index_failure))
        self.assertTrue(mb.is_damage(IntegrityError("immutable object hash or length mismatch: blob:00")))
        backend.arm_restore("svd_hevc")
        try:
            with mock.patch.object(backend, "_restore_lossy", side_effect=index_failure):
                with self.assertRaises(IntegrityError) as caught:
                    backend.restore(epoch)
        finally:
            backend.arm_restore(None)
        self.assertIs(caught.exception, index_failure)  # re-raised as is: not a damaged archive
        self.assertEqual(backend.archive_state(epoch, "svd_hevc"), "published")

    # --------------------------------------------------------------------------------- corrupt packet (RAW)
    def test_raw_packet_corruption_is_typed_with_verification_and_silent_without(self):
        from reelcache.errors import IntegrityError

        provider = self.provider(representation="raw")
        run_request(provider, {"session_id": "r", "action": "generate", "durable": True}, frames=initial_frames(4),
                    num_segments=2)
        epochs = segment_epochs(provider, "r")
        reference = conditioning_cache(provider.archive.frames(provider.archive.path(
            provider.archive.session("r")["head"])[1]), 1, "r")
        branch = mb.raw_branch(epochs[1])
        info = provider.runtime.branch_info(branch)
        corrupt(provider.runtime.store, blobs(provider.runtime.store, info["head_id"])[0])
        with self.assertRaises(IntegrityError) as caught:
            run_request(provider, {"session_id": "s", "action": "fork", "from_session": "r", "checkpoint": 1})
        self.assertEqual(code_of(caught.exception), "integrity")
        self.assertEqual(provider.backend.archive_state(epochs[1], "raw"), "quarantined")
        self.assertNotIn("s", provider.archive.session_ids())
        provider.close()
        # the old unverified reads restore the corrupt packet silently: different values, no error
        old = self.provider(representation="raw", verify_reads=False, on_open_error="raise")
        old.backend.arm_restore("raw")
        restored = old.backend.restore(epochs[1])
        old.backend.arm_restore(None)
        same = all(torch.equal(a, b) for a, b in zip(restored.keys_pre_rope + restored.values,
                                                      reference.keys_pre_rope + reference.values))
        self.assertFalse(same)
        old.close()
        # a damaged RAW chunk manifest is found when the store opens: the epoch's RAW archive is quarantined
        corrupt(old.runtime.store, info["chunk_manifests"][0])  # paths only: the store is closed
        again = self.provider(representation="raw")
        damage = again.startup["integrity"]["damage"]
        self.assertIn("branches/" + branch, damage["runtime_quarantined"])
        self.assertEqual(set(damage["raw_branches"]), {branch})
        self.assertEqual(again.backend.archive_state(epochs[1], "raw"), "quarantined")
        self.assertEqual(again.backend.archive_state(epochs[0], "raw"), "published")
        with self.assertRaisesRegex(IntegrityError, "quarantined"):
            run_request(again, {"session_id": "t", "action": "fork", "from_session": "r", "checkpoint": 1})
        ok = run_request(again, {"session_id": "u", "action": "fork", "from_session": "r", "checkpoint": 0})
        self.assertEqual(ok["summary"]["segments"][0]["provider_cache"], "restore")

    # -------------------------------------------------------------------------------- damage found at open
    def test_damaged_records_at_open_are_quarantined_and_other_sessions_work(self):
        from reelcache.errors import IntegrityError

        provider = self.provider()
        for sid, seed, segments in (("good", 10, 2), ("badrec", 11, 1), ("badchain", 12, 2), ("badlin", 13, 2),
                                    ("named", 14, 1)):
            run_request(provider, {"session_id": sid, "action": "generate", "durable": True},
                        frames=initial_frames(seed), num_segments=segments)
        provider.control("checkpoint", {"session_id": "named", "name": "x"})
        provider.control("rewind", {"session_id": "named", "checkpoint": 0})  # "x" is off the head path now
        good_epochs, lin_epochs = segment_epochs(provider, "good"), segment_epochs(provider, "badlin")
        good_head = provider.archive.session("good")["head"]
        store, archive = provider.runtime.store, provider.archive
        chain = archive.path(archive.session("badchain")["head"])
        named_ref = archive.named("named")["x"]
        record_ref = store.get_ref("longcat/sessions/badrec")
        lineage_head = provider.runtime.list_lineages()[mb.lineage_branch("badlin")]
        provider.close()
        for object_id in (record_ref, chain[1].ref, named_ref, lineage_head):
            corrupt(store, object_id)  # paths only: the store is closed
        provider = self.provider()  # the attach succeeds
        store = provider.runtime.store
        damage = provider.startup["integrity"]["damage"]
        self.assertEqual(set(damage["sessions"]), {"badrec", "badchain"})
        self.assertEqual(set(damage["checkpoints"]), {"longcat/checkpoints/named/x"})
        self.assertEqual(set(damage["runtime_quarantined"]), {"lineages/" + mb.lineage_branch("badlin")})
        self.assertEqual(set(damage["lineages"]), {mb.lineage_branch("badlin")})
        self.assertEqual({(item["session_id"], item["segment_index"]) for item in damage["epochs"]},
                         {("badlin", 0), ("badlin", 1)})
        listed = {entry["session_id"]: entry for entry in provider.control("list", {})["sessions"]}
        self.assertEqual({sid: entry["status"] for sid, entry in listed.items()},
                         {"good": "closed", "badrec": "quarantined", "badchain": "quarantined", "badlin": "closed",
                          "named": "closed"})
        self.assertIn("IntegrityError", listed["badrec"]["quarantined"])
        self.assertEqual(set(listed["badlin"]["damage"]), {"lineages", "epochs"})
        self.assertEqual(set(listed["named"]["damage"]), {"checkpoints"})
        # a quarantined session: describe reports it, every other op is a typed integrity error
        described = provider.control("describe", {"session_id": "badchain"})
        self.assertEqual((described["status"], described["segments"]), ("quarantined", []))
        for op, args in (("open", {}), ("pause", {}), ("checkpoints", {}), ("checkpoint", {"name": "y"}),
                         ("rewind", {"checkpoint": 0}), ("close", {}), ("fork", {"target": "z"})):
            self.assert_control_error(provider, "integrity", op, session_id="badrec", **args)
        from reelcache.integrations.sglang.controls import SessionControlError
        for index, handle in enumerate(({"session_id": "badchain", "action": "resume"},
                                        {"session_id": "z", "action": "fork", "from_session": "badrec",
                                         "checkpoint": 0})):
            with self.assertRaises(SessionControlError) as caught:
                run_request(provider, handle, request_id=f"refused-{index}")
            self.assertEqual(caught.exception.code, "integrity")
            refused = provider.control("request", {"request_id": f"refused-{index}"})["summary"]
            self.assertEqual((refused["refused"], refused["error_code"]), (True, "integrity"))
        self.assert_control_error(provider, "illegal_transition", "fork", session_id="good", target="badrec")
        # a session whose cache lineage is quarantined: typed errors, its records unchanged
        lin_head = provider.archive.session("badlin")["head"]
        with self.assertRaises(IntegrityError):
            run_request(provider, {"session_id": "badlin", "action": "resume"})
        with self.assertRaisesRegex(IntegrityError, "quarantined"):
            run_request(provider, {"session_id": "w", "action": "fork", "from_session": "badlin", "checkpoint": 1})
        self.assertEqual(provider.archive.session("badlin")["head"], lin_head)
        self.assertNotIn("w", provider.archive.session_ids())
        self.assertEqual({provider.backend.archive_state(epoch, "svd_hevc") for epoch in lin_epochs}, {"quarantined"})
        # the damaged named checkpoint is listed and typed; the session itself works
        named = provider.control("checkpoints", {"session_id": "named"})["named"]["x"]
        self.assertIn("IntegrityError", named["quarantined"])
        self.assert_control_error(provider, "integrity", "fork", session_id="named", target="v", checkpoint="x")
        # every other session works: fork restores the published epoch, resume builds
        forked = run_request(provider, {"session_id": "g2", "action": "fork", "from_session": "good", "checkpoint": 1})
        self.assertEqual(forked["summary"]["segments"][0]["provider_cache"], "restore")
        run_request(provider, {"session_id": "good", "action": "resume"})
        self.assertEqual({provider.backend.archive_state(epoch, "svd_hevc") for epoch in good_epochs}, {"published"})
        # garbage collection runs, keeps every RAW branch / holder while a session record is damaged, and keeps the
        # damaged objects
        result = provider.control("gc", {})["gc"]
        self.assertEqual(result["conservative"]["quarantined"], ["longcat/checkpoints/named/x",
                                                                 "longcat/sessions/badchain",
                                                                 "longcat/sessions/badrec"])
        for object_id in (chain[1].ref, named_ref, lineage_head):
            self.assertTrue(store.path_for(object_id).exists())
        self.assertNotEqual(provider.archive.session("good")["head"], good_head)  # resumed
        # deleting quarantined sessions works without reading them; the next collection is complete again
        deleted = provider.control("delete", {"session_id": "badrec"})
        # a store that quarantines objects which failed a check (freeze D core) re-raises its QuarantinedError
        self.assertRegex(deleted["quarantined"], r"IntegrityError|QuarantinedError")
        provider.control("delete", {"session_id": "badchain"})
        provider.control("delete", {"session_id": "named", "checkpoint": "x"})
        final = provider.control("gc", {})["gc"]
        self.assertIsNone(final["conservative"])
        self.assertEqual(set(provider.archive.session_ids()), {"good", "badlin", "named", "g2"})
        self.assertFalse(store.path_for(chain[1].ref).exists())  # unreachable once its sessions are gone
        provider.close()
        # the old policy refuses the damaged store at attach (typed)
        with self.assertRaises(IntegrityError):
            make_provider(self.root / "store", on_open_error="raise")

    # ------------------------------------------------------------------------- session-level CPU packing
    def test_prepare_isolates_and_quarantines_a_damaged_epoch(self):
        from reelcache.errors import IntegrityError

        provider = self.provider()
        run_request(provider, {"session_id": "p", "action": "generate", "durable": True}, frames=initial_frames(20),
                    num_segments=2)
        backend = provider.backend
        epochs = segment_epochs(provider, "p")
        damaged = backend.node_ref(epochs[1])
        prepared = []

        class Restorer:
            """A restorer that prepares node packs (reads them); the damaged node's pack fails verification."""

            def history(self, node_ref):
                return [{"node_id": node_ref}]

            def prewarm(self, items):
                if any(item["node_id"] == damaged for item in items):
                    raise IntegrityError(f"immutable object hash or length mismatch: pack of {damaged}")
                prepared.extend(item["node_id"] for item in items)
                return 0.0

        backend.restorer = Restorer()
        report = backend.prepare_epochs(epochs)
        self.assertEqual((report["quarantined"], prepared), ([1], [backend.node_ref(epochs[0])]))
        self.assertEqual([backend.archive_state(epoch, "svd_hevc") for epoch in epochs], ["published", "quarantined"])
        report = backend.prepare_epochs(epochs)  # quarantined epochs are skipped
        self.assertEqual((report["skipped_quarantined"], report["quarantined"]), ([1], []))
        backend.integrity = {**backend.integrity, "on_open_error": "raise"}
        backend._index[mb.epoch_key(epochs[1])].quarantined.clear()
        with self.assertRaises(IntegrityError):
            backend.prepare_epochs(epochs)


if __name__ == "__main__":
    unittest.main()
