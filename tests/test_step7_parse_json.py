"""Step 7: ``serialization.parse_json`` fast path (one ``json.loads`` with validating hooks, no ``json_copy`` pass)
accepts and rejects exactly what the former implementation did and returns equal values.

The former implementation is kept below verbatim as the reference. Inputs: every manifest of committed lineages
(nodes, epochs, records, packs, basis versions), adversarial documents (duplicate keys, NaN/Infinity constants,
out-of-range number literals, huge integers, invalid UTF-8, truncation, BOM, ...), seeded random mutations, and
nesting depths around the recursion boundary (the former validation was recursive).
"""

from __future__ import annotations

import json
import math
import random
import unittest
from typing import Any, Mapping

from reelcache.errors import ConfigurationError, IntegrityError
from reelcache.lineage_commit import ArchiveOptions
from reelcache.serialization import canonical_json, parse_json
from tests import step6_support as s6


# -- the former implementation (reference, verbatim) -----------------------------------------------------------
def _reference_json_copy(value: Any, *, name: str = "metadata") -> Any:
    """Validate JSON without accepting NaNs, objects, or non-string keys."""
    def check(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise ConfigurationError(f"{name} keys must be strings")
            for child in item.values():
                check(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                check(child)
            return
        raise ConfigurationError(f"{name} contains a non-JSON value: {type(item).__name__}")

    def plain(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        return item

    try:
        check(value)
        return json.loads(json.dumps(plain(value), allow_nan=False))
    except (RecursionError, ValueError, OverflowError) as exc:
        raise ConfigurationError(f"{name} is not a finite acyclic JSON value") from exc


def _reference_parse_json(data: bytes) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise IntegrityError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique_pairs)
        return _reference_json_copy(value)
    except (ValueError, UnicodeError, TypeError, RecursionError, ConfigurationError) as exc:
        raise IntegrityError("invalid JSON metadata") from exc


# -- comparison ----------------------------------------------------------------------------------------------
def _outcome(function, data):
    try:
        return ("ok", function(data))
    except Exception as exc:  # noqa: BLE001 - compared below
        return ("error", type(exc), str(exc))


def _same_value(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return list(a) == list(b) and all(_same_value(a[key], b[key]) for key in a)
    if type(a) is list:
        return len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
    if type(a) is float:
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return a == b


class ParseJSONEquivalenceTests(unittest.TestCase):
    def assertSameOutcome(self, data):
        expected, actual = _outcome(_reference_parse_json, data), _outcome(parse_json, data)
        self.assertEqual(expected[0], actual[0], (data[:120], expected, actual))
        if expected[0] == "ok":
            self.assertTrue(_same_value(expected[1], actual[1]), data[:120])
        else:
            self.assertEqual(expected[1:], actual[1:], data[:120])
        return actual[0]

    def test_every_stored_manifest_parses_identically(self):
        runtime = s6.RuntimeFactory(self)("parse-json")
        registry = s6.registry()
        runtime.create_lineage("main", streams=())
        for index in range(3):
            runtime.commit_node("main", f"chunk-{index:03d}", *s6.chunk(index), method=s6.method(),
                                registry=registry, options=ArchiveOptions(encode_threads=1))
        runtime.create_lineage("cfg", streams=("cond", "uncond"))
        for index in range(2):
            runtime.commit_node("cfg", f"chunk-{index:03d}", *s6.chunk(index, streams=("cond", "uncond")),
                                method=s6.method(), registry=registry, options=ArchiveOptions(encode_threads=1))
        store = runtime.store
        ids = [row[0] for row in store._db.execute("SELECT id FROM objects WHERE id LIKE 'manifest:%'")]
        self.assertGreater(len(ids), 20)
        for object_id in ids:
            data = store._get(object_id)
            self.assertEqual(self.assertSameOutcome(data), "ok", object_id)
            self.assertEqual(parse_json(canonical_json(parse_json(data))), parse_json(data))

    def test_adversarial_documents(self):
        huge = "9" * 5000
        cases = [
            b'{"a": 1, "a": 2}', b'{"a": {"b": 1, "b": 1}}', b'[{"x": 1}, {"y": 2, "y": 3}]',
            b'{"a": 1, "b": {"c": 1, "c": 2}, "a": 3}', b'{"k": 1, "K": 2}',
            b'NaN', b'[NaN]', b'{"a": Infinity}', b'{"a": -Infinity}', b'[1, 2, {"b": [NaN]}]',
            b'1e400', b'-1e400', b'[1E999]', b'{"a": 1.7976931348623157e309}', b'{"a": 1.7976931348623157e308}',
            b'1e-400', b'-0.0', b'-0', b'0', b'1.0', b'1', b'[1.5e-320]', b'{"a": 123456789012345678901234567890}',
            huge.encode(), ("[" + huge + "]").encode(), ("1." + huge).encode(), ("1e" + huge).encode(),
            b'"\\ud800"', b'"\\u0000"', '"\u00e9\u4e2d"'.encode(), b'"\xff"', b'\xff\xfe', b'\xef\xbb\xbf{}',
            b'', b' ', b'{', b'[1, 2', b'{"a": 1} x', b'{"a": 1,}', b'[1,]', b"{'a': 1}", b'{"a" 1}', b'nul',
            b'null', b'true', b'false', b'[]', b'{}', b'[[]]', b'[{}]', b'{"": {"": []}}',
            b'{"a": [1, 2.5, "x", null, true, false, {"b": -0.0}]}', b'  {"a":1}  \n',
            b'[0x10]', b'[01]', b'[+1]', b'[.5]', b'[1.]', b'["\\x41"]', b'["\\u12"]', b'["a\tb"]',
        ]
        for data in cases:
            with self.subTest(data=data[:40]):
                self.assertSameOutcome(data)
        with self.assertRaisesRegex(IntegrityError, "duplicate JSON key: a"):
            parse_json(b'{"a": 1, "a": 2}')
        for data in (b"[NaN]", b"1e400", b'{"a": -Infinity}'):
            with self.assertRaisesRegex(IntegrityError, "invalid JSON metadata"):
                parse_json(data)
        for data in ("{}", None, 1):  # not bytes: the same (unwrapped) error as before
            with self.subTest(data=data):
                expected, actual = _outcome(_reference_parse_json, data), _outcome(parse_json, data)
                self.assertEqual(expected[:2], actual[:2])

    def test_seeded_random_documents_and_mutations(self):
        rng = random.Random(20261016)

        def document(depth=0):
            kind = rng.random()
            if depth > 4 or kind < 0.35:
                return rng.choice([None, True, False, rng.randint(-10**6, 10**6), rng.uniform(-1e6, 1e6),
                                   rng.choice(["", "a", "é", "\\", '"q"']), rng.random() * 1e-300, -0.0])
            if kind < 0.65:
                return [document(depth + 1) for _ in range(rng.randint(0, 4))]
            return {f"k{rng.randint(0, 6)}": document(depth + 1) for _ in range(rng.randint(0, 4))}
        mutations = [
            lambda t: t, lambda t: t.replace("1", "NaN", 1), lambda t: t.replace("0", "Infinity", 1),
            lambda t: t.replace(".", "e999", 1), lambda t: t.replace("e-", "e+", 1),
            lambda t: t.replace('"k1"', '"k2"', 1), lambda t: t.replace('{"k', '{"k0": 1, "k0": 2, "k', 1),
            lambda t: t[:-1], lambda t: t + "]", lambda t: t.replace(":", ",", 1),
        ]
        outcomes = {"ok": 0, "error": 0}
        for _ in range(600):
            text = json.dumps(document(), ensure_ascii=bool(rng.getrandbits(1)))
            data = rng.choice(mutations)(text).encode("utf-8")
            outcomes[self.assertSameOutcome(data)] += 1
        self.assertGreater(outcomes["ok"], 100)
        self.assertGreater(outcomes["error"], 100)

    def test_nesting_depth_boundary_is_unchanged(self):
        shapes = {
            "arrays": lambda depth: "[" * depth + "]" * depth,
            "arrays-with-scalar": lambda depth: "[" * depth + "1" + "]" * depth,
            "objects": lambda depth: '{"a":' * depth + "1" + "}" * depth,
            "empty-object-leaf": lambda depth: '{"a":' * depth + "{}" + "}" * depth,
            "mixed": lambda depth: '[{"a":' * (depth // 2) + "[1, {}]" + "}]" * (depth // 2),
        }
        for name, build in shapes.items():
            with self.subTest(shape=name):
                def accepted(depth, function):
                    return _outcome(function, build(depth).encode())[0] == "ok"
                low, high = 1, 4000  # the reference accepts low and rejects high
                self.assertTrue(accepted(low, _reference_parse_json))
                self.assertFalse(accepted(high, _reference_parse_json))
                while high - low > 1:
                    middle = (low + high) // 2
                    if accepted(middle, _reference_parse_json):
                        low = middle
                    else:
                        high = middle
                for depth in (low - 1, low, high, high + 1):
                    self.assertEqual(accepted(depth, parse_json), accepted(depth, _reference_parse_json),
                                     (name, depth))
                # Around the fast path's own threshold (64 levels): both accept, with equal values.
                for depth in (62, 63, 64, 65, 66, 67, 130):
                    data = build(depth).encode()
                    self.assertTrue(_same_value(parse_json(data), _reference_parse_json(data)), (name, depth))


if __name__ == "__main__":
    unittest.main()
