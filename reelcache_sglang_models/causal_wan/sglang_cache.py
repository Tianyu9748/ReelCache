"""LongLive 1.0 KV cache with the upstream prompt-switch recompute rule (imports SGLang).

SGLang's ``CausalSelfAttentionKVCache`` already implements LongLive's generation-time ring buffer exactly
(3-frame sink, rolling window, eviction of the oldest non-sink tokens, same local/global index arithmetic).
The only missing semantics is the recompute write of a KV-recache: upstream
``LongLive1.0/wan/modules/causal_model.py:228-302`` sets ``is_recompute = current_end <= global_end_index and
current_start > 0`` and, unless ``sink_recache_after_switch`` (global_sink false), starts the write at
``max(local_start, sink_tokens)`` so the sink keeps its KV; indices are not moved. SGLang would overwrite the
sink with the recomputed window. This subclass applies the upstream rule only in that case (a recompute span
that reaches into the sink); every other call is SGLang's own code path.

The SGLang branch ``reelcache-integration`` (Step 11, change request b) adds the same rule to the base cache as the opt-in
field ``protect_sink_on_recompute``; the profile stage uses that field when it exists and this subclass only on
e00930c (``sglang_stages.BASE_CACHE_PROTECTS_SINK``).

Resident session KV (branch commit 1663578341): ``CausalSelfAttentionKVCache.mutation_generation`` is a process-unique
tag renewed by every cache method that writes K/V, and ``index_state`` / ``holds_index_state`` /
``restore_index_state`` let the ReelCache bridge continue a session on the KV still resident in a stage's caches when
nothing wrote them since. Profile code that writes ``k``/``v`` directly (outside those methods) calls
:func:`note_kv_write` so the tag stays truthful.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalAttentionKVView,
    CausalSelfAttentionKVCache,
)

try:  # branch reelcache-integration >= 1663578341: the K/V mutation tag of the resident-KV seam
    from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import _next_kv_mutation
except ImportError:  # e00930c: caches carry no mutation tag (and no bridge can reuse them)
    _next_kv_mutation = None

#: SGLang's caches can tell a backend whether their K/V changed since a request ended (resident session KV).
RESIDENT_KV_SEAM = _next_kv_mutation is not None and all(
    callable(getattr(CausalSelfAttentionKVCache, name, None))
    for name in ("index_state", "holds_index_state", "restore_index_state"))


def note_kv_write(cache) -> None:
    """Renew ``cache.mutation_generation`` after writing its ``k``/``v`` outside SGLang's cache methods.

    Uses SGLang's own process-wide counter, so the new tag never equals a tag taken earlier from any cache (an
    ``index_state`` recorded before the write is no longer held). No-op on trees without the tag."""
    if _next_kv_mutation is not None and hasattr(cache, "mutation_generation"):
        cache.mutation_generation = _next_kv_mutation()


@dataclass(slots=True)
class RecacheAwareCausalKVCache(CausalSelfAttentionKVCache):
    # True only during a recache when global_sink is false (upstream sink_recache_after_switch): the sink is
    # recomputed too.
    sink_recache_after_switch: bool = False
    protected_recompute_writes: int = 0

    def update_and_get_attention_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        current_chunk_start: int,
        cache_head_start: int | None = None,
        recent_window_tokens: int | None = None,
        debug_name: str = "causal KV cache",
    ) -> CausalAttentionKVView:
        num_new = key.shape[1]
        global_end, local_end_prev = self._read_indices()
        current_end = current_chunk_start + num_new
        sink = self._effective_sink_tokens()
        is_recompute = current_end <= global_end and current_chunk_start > 0
        window_start = global_end - local_end_prev
        local_start = current_chunk_start - window_start
        protect = (is_recompute and not self.sink_recache_after_switch and sink > 0 and local_start < sink
                   and recent_window_tokens is None and self.num_cache_heads == key.shape[2]
                   and not self._has_pinned_sink() and self.global_sink_tokens == 0)
        if not protect:
            return super(RecacheAwareCausalKVCache, self).update_and_get_attention_kv(
                key=key, value=value, current_chunk_start=current_chunk_start, cache_head_start=cache_head_start,
                recent_window_tokens=recent_window_tokens, debug_name=debug_name)
        local_end = local_start + num_new
        if local_start < 0 or local_end > self.cache_size:
            raise RuntimeError(f"Invalid {debug_name} recompute range: local=[{local_start}, {local_end}), "
                               f"global_end={global_end}, prev_local_end={local_end_prev}, cache={self.cache_size}")
        write_start = max(local_start, sink)
        offset = write_start - local_start
        note_kv_write(self)  # like every base write path (resident-KV seam)
        if write_start < local_end:
            self.k[:, write_start:local_end] = key[:, offset:]
            self.v[:, write_start:local_end] = value[:, offset:]
        self.protected_recompute_writes += 1
        attn_start_index = max(0, local_end_prev - self.attention_window_size)
        visible_k, visible_v = self._visible_attention_kv(
            local_start_index=local_start, updated_local_end=local_end_prev, attn_start_index=attn_start_index,
            recent_window_tokens=None)
        return CausalAttentionKVView(k=visible_k, v=visible_v, local_start_index=local_start, local_end_index=local_end,
                                     visible_local_end=local_end_prev, visible_global_end=global_end)
