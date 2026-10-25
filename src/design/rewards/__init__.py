"""Fixed standalone rewards with lazy model exports."""

__all__ = ["ClashReward", "ClashScoreBatch"]

def __getattr__(name):
    if name in __all__:
        from .clash import ClashReward, ClashScoreBatch
        return {"ClashReward": ClashReward, "ClashScoreBatch": ClashScoreBatch}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
