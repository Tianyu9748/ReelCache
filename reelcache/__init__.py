"""ReelCache: immutable clean-chunk state for autoregressive video models."""

__version__ = "0.1.0"

__all__ = [
    "__version__", "RankPolicy", "RankDecision", "RankState",
    "RankPreparation", "PreparedRankMatrix", "RankAbstention",
]


def __getattr__(name):
    if name in __all__ and name != "__version__":
        from . import online_rank
        return getattr(online_rank, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
