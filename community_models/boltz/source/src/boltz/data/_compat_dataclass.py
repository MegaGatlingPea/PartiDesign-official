"""Dependency-absent serialization for the dataclass shapes in Boltz records.

This is the small subset of mashumaro used by Record and Manifest, not a general
replacement. Unknown keys are ignored, as with the default DataClassDictMixin;
required fields, nested dataclasses and declared container shapes are checked.
"""

from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from functools import lru_cache
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints


@lru_cache(maxsize=None)
def _field_types(cls: type) -> dict:
    return get_type_hints(cls)


def _dataclass_dict(value: object) -> dict:
    cls = type(value)
    annotations = _field_types(cls)
    return {
        field.name: _convert(getattr(value, field.name), annotations[field.name], dump=True)
        for field in fields(cls)
    }


def _dataclass_from_dict(cls: type, value: object) -> object:
    if not isinstance(value, Mapping):
        msg = f"Expected a mapping for {cls.__name__}"
        raise TypeError(msg)

    annotations = _field_types(cls)
    kwargs = {}
    for field in fields(cls):
        if not field.init:
            continue
        if field.name in value:
            try:
                kwargs[field.name] = _convert(
                    value[field.name], annotations[field.name], dump=False
                )
            except (TypeError, ValueError) as exc:
                msg = f"Invalid value for {cls.__name__}.{field.name}: {exc}"
                raise TypeError(msg) from exc
        elif field.default is MISSING and field.default_factory is MISSING:
            msg = f"Missing required field {cls.__name__}.{field.name}"
            raise TypeError(msg)
    return cls(**kwargs)


def _convert(value: object, annotation: object, *, dump: bool) -> object:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        # Preserve the representation of Union[str, int] identifiers.
        ordered = sorted(args, key=lambda item: type(value) is not item)
        for item in ordered:
            try:
                return _convert(value, item, dump=dump)
            except (TypeError, ValueError):
                pass
        msg = f"Expected {annotation}, got {type(value).__name__}"
        raise TypeError(msg)

    if annotation is type(None):
        if value is None:
            return None
    elif is_dataclass(annotation):
        if not dump:
            return _dataclass_from_dict(annotation, value)
        if isinstance(value, annotation):
            return _dataclass_dict(value)
    elif origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            msg = f"Expected a list or tuple, got {type(value).__name__}"
            raise TypeError(msg)
        if origin is list:
            return [_convert(item, args[0], dump=dump) for item in value]
        if len(args) != len(value):
            msg = f"Expected a tuple of length {len(args)}, got {len(value)}"
            raise ValueError(msg)
        converted = [
            _convert(item, item_type, dump=dump)
            for item, item_type in zip(value, args, strict=True)
        ]
        return converted if dump else tuple(converted)
    elif annotation is bool:
        if isinstance(value, bool):
            return bool(value)
    elif annotation is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
    elif annotation is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif annotation is str:
        if isinstance(value, str):
            return str(value)
    else:
        msg = f"Unsupported record field type: {annotation}"
        raise TypeError(msg)

    msg = f"Expected {annotation}, got {type(value).__name__}"
    raise TypeError(msg)


class DataClassDictMixin:
    """Provide the dictionary methods used by Boltz JSONSerializable."""

    def to_dict(self) -> dict:
        """Convert nested record dataclasses to JSON-compatible dictionaries."""
        return _dataclass_dict(self)

    @classmethod
    def from_dict(cls: type["DataClassDictMixin"], value: dict) -> "DataClassDictMixin":
        """Restore nested record dataclasses, retaining their declared defaults."""
        return _dataclass_from_dict(cls, value)
