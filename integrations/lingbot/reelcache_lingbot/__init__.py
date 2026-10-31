"""ReelCache integration for LingBot-World-Fast (14B, 40 layers; official repo Robbyant/lingbot-world @ a43bec7).

LingBot runs outside SGLang: :mod:`.lb_model` drives the official pipeline chunk by chunk through the runtime wrapper
``integrations/lingbot/runtime/lingbot_rt.py`` (hooks only, no source edits), :mod:`.lb_archive` archives and restores
its KV with the ReelCache runtime (``reelcache.integrations.sglang.lineage_archive.LineageSessionArchive``), and
:mod:`.lb_overlap` overlaps a history restore layer by layer with the first forward of the next chunk.
"""
