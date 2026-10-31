# Provenance

The paper's runtime was developed under an internal working name. This release is that code with the name replaced
by ReelCache everywhere: packages, modules, classes, functions, CLI flags, environment variables, file names,
documentation, and format identifiers such as `reelcache.tensor.v1` and the tensor-file magic. The paths of the
original cluster were then replaced by placeholders and settings
([machine-specific paths](installation.md#machine-specific-paths)); the tests that compare against evaluation
artifacts read them from `REELCACHE_TEST_ASSETS` ([test assets](testing.md#test-assets)).

- **Runtime** (`reelcache/`, `reelcache_sglang_models/`, `native/`, `examples/`, `scripts/`, `tests/`, `pyproject.toml`).
  - Source: the composed tree `ALL_v1` (tree id `b1f55ab7573dd856`). That tree is the paper's W1.4.3 runtime freeze
    (`63b731cf19f6b384`), plus two opt-in restore flags (`pipelined_restore`, and `verify_bases=false` to skip
    hashing) and the [acceleration levers](configuration.md#acceleration-levers). With every new flag at its default, its archives (every stored object
    kind) and generated latents are bit-identical to that freeze. This was checked on GPU for CausVid and RAVEN.
  - Checked: applying the rename to `ALL_v1` reproduces 278 of its 298 code files byte for byte; its prebuilt `.so`
    is not shipped. The other 20 differ only by these hand edits.
    - For the rename:
      - `pyproject.toml`: the distribution name.
      - Six test modules: hashes they pin that the rename changes. Each new value is proven to be the digest of the
        same input with the renamed identifiers.
      - `test_generation_first.py`, `test_wl_detach.py`: they compare against renamed copies of the frozen W1.4.1
        and W1.2 trees.
      - `test_step4_real_data.py`: it reads converted copies of the real CausVid captures, with bit-identical tensors.
      - `test_step7_store_faults.py`: one test's corruption is pinned to a structural byte. It used to flip the
        manifest's middle byte, which the longer identifiers moved.
    - For the cluster paths:
      - `causal_wan/profiles.py` (`MODEL_ROOT`), `longcat_video/weights.py` (`DEFAULT_CHECKPOINT_DIR`) and
        `examples/benchmark_interactive_sessions.py` (no default runner root).
      - `test_wl_prompt_switch.py`: the profile identities it pins include the bound weight paths. Each new value is
        proven to be the digest of the same input with the paths mapped to `/models`, nested fingerprints recomputed.
      - `test_generation_first.py`, `test_wl_detach.py`, `test_causal_wan_profiles.py`, `test_longcat_video_port.py`,
        `test_step4_real_data.py`: they find their artifacts through `tests/assets.py` (new);
        `test_longcat_video_context_parallel.py` skips its Gloo tests, whose workers load the LongCat-Video source,
        when that asset is not configured.
      - `test_causal_wan_profiles.py`: the profile-directory tampering test binds stand-in component files, not the
        Diffusers snapshot on disk.
      - `test_sglang_bridge.py`, `test_vbench_causvid.py`, `test_step7_cpu_pool.py`: synthetic paths, a mocked mount
        point and a comment.
  - Content ids differ from the evaluation's only through the renamed identifiers and, for the causal-Wan profiles,
    the placeholder weight paths (a profile's fingerprint includes them). Stores written by the evaluated runtime are
    not readable by this release.
- **SGLang:** the evaluated SGLang commit `043c0204cffef4c75262ec7e5ae2a2141645fe85` is upstream `e00930c5`
  (0.5.21) plus 26 commits. The patches here are those commits under the same rename, plus one test line (see
  [`third_party/sglang/README.md`](../third_party/sglang/README.md)). They apply to give git tree `06209d51…`.
- **LingBot integration:** the runtime wrapper and the model/archive/overlap adapters used in the evaluation. The
  edits are import paths, environment variables with placeholder defaults in place of cluster paths, the shipped
  config template and the rename (see its README).
- **This repository's checks** (final tree):
  - CPU suite with the [test assets](testing.md#test-assets): 2,431 tests, 0 failures, 0 errors, 0 skips (`scripts/verify.py
    --require-no-skips` on a node-local copy, with the Rust extension built from `native/reelcache_native`;
    CPython 3.12.13, PyTorch 2.8, CPU only), plus the LingBot tests (9/9).
  - The same without the test assets, as in a fresh clone: 2,424 tests run, 0 failures, 0 errors, 29 skipped, all of
    them the asset-dependent tests of the six modules listed in [testing](testing.md#test-assets) (a class skipped in its setup counts once). LingBot: 8
    passed, 1 skipped.
  - SGLang unit tests (seam tests plus related upstream modules, CPU), renamed SGLang with this repository versus
    the evaluated SGLang with `ALL_v1`: 632 passed, 33 failed, 40 skipped in both, with identical failing tests.
    None of the 12 seam test modules fails. The 33 failures are upstream `test_server_args.py` cases that depend
    on the machine (offload placement decided without a GPU; a hard-coded `/root/.cache` path).
  - GPU smoke (one RTX PRO 6000, serving environment): the 20 test modules with CUDA-dependent paths, 413 tests,
    run with the GPU visible on this repository and on `ALL_v1`. Results are identical: 409 passed, 1 skipped, and
    the same 3 tests fail in both. Two are LongCat-Video session-provider tests whose eager-overlap memory check
    finds no room on this node (43 GiB needed, 0 GiB counted as available). The third asserts that CUDA is not yet
    initialized, which depends on test order within one process.

## What is not included

- Experiment harnesses and runners: the interactive session runner, the sustained-overlap and acceleration
  benchmarks, the LingBot evaluation runner, Slurm jobs.
- Workload bundles, authored prompt banks and datasets.
- Evaluation results, figures, decks, reports and the paper sources.
- Frozen runtime trees, model weights, stores and any built binaries (`.so`, wheels).

## Licensing

The packages here declare no license. The SGLang patches derive from SGLang (Apache-2.0); the Rust crate's
manifest declares MIT.
