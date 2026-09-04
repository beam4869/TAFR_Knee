"""Package-specific exceptions."""


class TAFRKneeError(Exception):
    """Base class for TAFR-Knee errors."""


class ConfigurationError(TAFRKneeError, ValueError):
    """Raised when an algorithm or problem configuration is inconsistent."""


class SolveError(TAFRKneeError, RuntimeError):
    """Raised when a scalarized lower-level solve fails."""
