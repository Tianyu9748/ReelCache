# ReelCache documentation

| Document | Contents |
|---|---|
| [design.md](design.md) | How ReelCache works: capture, compression, basis sharing, encoding, storage, tiers, restore, interference control and the built-in baselines; the supported models |
| [repository-structure.md](repository-structure.md) | The repository tree and the purpose of every package and module |
| [installation.md](installation.md) | Dependencies with their evaluated versions; installing the runtime, SGLang Diffusion with the patches and the Rust extension; model weights; machine-specific paths |
| [usage.md](usage.md) | Quick start without a GPU; serving with SGLang Diffusion and the session API; LingBot-World-Fast |
| [configuration.md](configuration.md) | `ReelCacheRuntimeConfig` fields, the evaluated configuration and the acceleration levers |
| [testing.md](testing.md) | The CPU test suite, its requirements and the optional test assets |
| [provenance.md](provenance.md) | Where the code comes from, how this release was checked, what is not included, licensing |

Documentation that lives next to the code:

| Document | Contents |
|---|---|
| [`reelcache/integrations/sglang/README.md`](../reelcache/integrations/sglang/README.md) | The SGLang bridge: publication across requests, resident KV, the seam protocol, session controls, profiles |
| [`reelcache_sglang_models/longcat_video/DESIGN.md`](../reelcache_sglang_models/longcat_video/DESIGN.md) | The LongCat-Video port |
| [`integrations/lingbot/README.md`](../integrations/lingbot/README.md), [`runtime/API.md`](../integrations/lingbot/runtime/API.md) | The LingBot-World-Fast integration and its runtime wrapper |
| [`third_party/sglang/README.md`](../third_party/sglang/README.md) | The 26 SGLang patches |
| [`native/reelcache_native/README.md`](../native/reelcache_native/README.md) | The optional Rust extension |
