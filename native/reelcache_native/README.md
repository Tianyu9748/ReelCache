# reelcache_native: optional Rust extension

A small PyO3 (0.23.5) extension. It moves three host-side hot spots of archival out of Python and releases the
GIL while they run:
- `start_codes(data)`: offsets of every `00 00 01` start code in an HEVC Annex-B stream (memchr);
- `nal_units(data)`: the NAL unit table (offsets, lengths, NAL types, first-slice flags), with the Python
  reference's integrity checks;
- `unpack_tile(...)`: copies one coefficient tile out of a decoded uint8 canvas `[pictures, planes, height, width]`
  into `[frames, h, w, rank]` (the per-cell unmosaic).

Every function returns exactly what its Python reference (`reelcache._hevc`, `reelcache.codecs`) returns.

Runtime use is opt-in: `ReelCacheRuntimeConfig(native_ext=True)`. When the module is missing or fails to load,
`reelcache.accel_c` falls back to its Python implementations. Both paths give identical outputs, which
`tests/test_accel_c_native.py` checks on real streams. The measured gain is about 53 ms of host time per archived
node, with no effect on generation speed, because the remaining interference is GPU-side.

## Build

Requirements: Rust 1.92 (any recent stable should work), `maturin>=1.5,<2`, and the CPython 3.12 interpreter
that will import it. Build outside the source tree, then place the module inside the `reelcache` package, where the
loader looks for it (`reelcache/_native/`):

```bash
cd native/reelcache_native
CARGO_TARGET_DIR=/tmp/reelcache-native-target \
  maturin build --release --locked -i "$(which python3.12)" -o /tmp/reelcache-native-wheels
python - <<'EOF'
import glob, zipfile, pathlib
wheel = glob.glob("/tmp/reelcache-native-wheels/reelcache_native-*.whl")[0]
name = "reelcache_native/reelcache_native.cpython-312-x86_64-linux-gnu.so"
pathlib.Path("../../reelcache/_native", pathlib.Path(name).name).write_bytes(zipfile.ZipFile(wheel).read(name))
EOF
cd ../.. && python -c "import reelcache._native.reelcache_native as m; print(m.__file__)"
```

`Cargo.lock` pins the dependency versions, so `--locked` builds offline once the crates are in the Cargo cache.
The built `.so` is git-ignored: the repository ships no binaries.
