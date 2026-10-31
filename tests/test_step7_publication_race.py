"""Step 7: a cancellation racing an in-flight publication never reports a published chunk as cancelled.

The publisher's compare-and-swap of the lineage head is irrevocable: a chunk inside it lands. Before the fix,
``cancel`` (or any failure) resolved that chunk's ticket with the cancellation while the publication then advanced
the head, so the SGLang bridge reported a published commit as retired (seen as a flaky
``test_close_abort_retires_unpublished_commits_and_drain_publishes_them`` under load). Now the in-flight
publication decides that chunk; every later one is cancelled.
"""

from __future__ import annotations

import threading
import time
import unittest

from reelcache.archive_pipeline import ArchivePipeline
from reelcache.lineage_commit import ArchiveOptions
from tests import step6_support as s6


class PublicationRaceTests(unittest.TestCase):
    def test_cancel_during_publication_lets_the_chunk_land_and_cancels_the_rest(self):
        runtime = s6.RuntimeFactory(self)("publication-race")
        runtime.create_lineage("main")
        entered, release = threading.Event(), threading.Event()

        def hooks(stage, node):
            if stage == "publish" and node == "chunk-000":
                entered.set()
                if not release.wait(60):
                    raise RuntimeError("test gate was not released")
        pipeline = ArchivePipeline(runtime, method=s6.method(), registry=s6.registry(),
                                   options=ArchiveOptions(encode_threads=1), hooks=hooks, encode_workers=2,
                                   max_source_chunks=3, max_unpublished=4)
        self.addCleanup(lambda: pipeline.close(drain=False, timeout=60))
        tickets = [pipeline.submit("main", f"chunk-{index:03d}", *s6.chunk(index)) for index in range(3)]
        self.assertTrue(entered.wait(120))  # chunk 0 is inside its publication
        cancelled = {}
        canceller = threading.Thread(target=lambda: cancelled.setdefault("head", pipeline.cancel("main",
                                                                                                timeout=120)))
        canceller.start()
        time.sleep(0.3)  # the cancellation has marked the lineage while chunk 0 publishes
        release.set()
        canceller.join(120)
        self.assertFalse(canceller.is_alive())
        self.assertIsNone(tickets[0].error)  # landed: reported as published, not as cancelled
        self.assertEqual(tickets[0].result(), runtime.lineage_head("main"))
        self.assertEqual(cancelled["head"], tickets[0].result())
        for ticket in tickets[1:]:
            self.assertTrue(ticket.done())
            self.assertIsNotNone(ticket.error)
        stats = pipeline.stats()
        self.assertEqual((stats["published"], stats["cancelled"]), (1, 2))


if __name__ == "__main__":
    unittest.main()
