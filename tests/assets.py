"""Test assets that are not part of this repository.

A few tests compare against artifacts of the original evaluation: copies of two earlier runtime trees, weight-hash
receipts, upstream model sources and checkpoints, and real KV captures. Point ``REELCACHE_TEST_ASSETS`` at a directory
with the layout in README section 9 to run them; without it they are skipped. A configured directory that lacks a file
a test needs fails that test.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

ENV = "REELCACHE_TEST_ASSETS"


def asset(*parts: str, override: str | None = None) -> Path:
    """The asset's path: the ``override`` environment variable when it is set (it names this one asset), else
    ``$REELCACHE_TEST_ASSETS/<parts>``; raises ``unittest.SkipTest`` when neither is configured. Call it inside a test,
    never at import."""
    if override and os.environ.get(override):
        return Path(os.environ[override])
    base = os.environ.get(ENV)
    if not base:
        raise unittest.SkipTest(f"needs {ENV} (test assets of the original evaluation; README section 9)")
    return Path(base).joinpath(*parts)
