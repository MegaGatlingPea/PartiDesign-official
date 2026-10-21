class PipeToolsError(RuntimeError):
    """Base class for user-facing pipeline failures."""


class ConfigError(PipeToolsError):
    """Configuration is invalid or inconsistent with an input."""


class StructureError(PipeToolsError):
    """A structure cannot be converted or interpreted safely."""


class AdapterError(PipeToolsError):
    """An external model adapter failed."""


class MetricError(PipeToolsError):
    """A requested structural metric is not evaluable."""

