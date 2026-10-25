"""Token padding policy shared by standalone compile configuration and collation."""

from collections.abc import Mapping, Sequence
from typing import Any

from .records import DesignConfigError, DesignInputError


DEFAULT_TOKEN_BUCKETS = (
    128, 160, 192, 224, 256, 288, 320, 352, 384,
    448, 512, 640, 768, 896, 1024,
)


def validate_token_buckets(value: Any) -> tuple[int, ...] | None:
    """Accept only null or a strictly increasing, nonempty list of positive ints."""
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise DesignConfigError("model.compile.token_buckets must be null or a nonempty list of positive integers")
    if any(type(bucket) is not int or bucket <= 0 for bucket in value):
        raise DesignConfigError("model.compile.token_buckets entries must be positive integers (not booleans)")
    if any(left >= right for left, right in zip(value, value[1:])):
        raise DesignConfigError("model.compile.token_buckets must be strictly increasing")
    return tuple(value)


def validate_bucket_compile_policy(config: Mapping[str, Any]) -> tuple[int, ...] | None:
    buckets = validate_token_buckets(config.get("token_buckets"))
    if config["dynamic"] is not (buckets is None):
        raise DesignConfigError(
            "model.compile requires dynamic=true with token_buckets=null, "
            "or dynamic=false with a nonempty token_buckets list"
        )
    return buckets


def validate_compile_enabled(value: Any) -> bool | str:
    if type(value) is bool or (type(value) is str and value == "auto"):
        return value
    raise DesignConfigError("model.compile.enabled must be boolean or 'auto'")


def active_token_buckets(config: Mapping[str, Any]) -> tuple[int, ...] | None:
    """Disable input padding with compilation while retaining the configured policy."""
    buckets = validate_bucket_compile_policy(config)
    enabled = validate_compile_enabled(config["enabled"])
    return buckets if enabled is True else None


def token_bucket_size(token_count: int, buckets: Sequence[int]) -> int:
    validated = validate_token_buckets(list(buckets))
    assert validated is not None
    for bucket in validated:
        if token_count <= bucket:
            return bucket
    raise DesignInputError(
        f"batch maximum token count {token_count} exceeds the largest "
        f"model.compile.token_buckets value {validated[-1]}; extend the bucket list "
        "or select the dynamic compile strategy"
    )
