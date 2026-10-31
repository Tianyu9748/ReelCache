"""Explicit failure modes for storage, representations, and native state."""


class ReelCacheError(Exception):
    """Base class for all declared runtime failures."""


class ConfigurationError(ReelCacheError, ValueError):
    """A configuration or identity is invalid or incompatible."""


class IntegrityError(ReelCacheError):
    """An immutable object is corrupt, missing, or internally inconsistent."""


class QuarantinedError(IntegrityError):
    """A head found damaged when its store was opened (``on_open_error="quarantine"``) was used."""


class CapacityError(ReelCacheError):
    """A bounded resource cannot admit the requested operation."""


class StateError(ReelCacheError):
    """The operation is not valid in the current runtime state."""


class UnsupportedError(ReelCacheError):
    """A requested representation or capability is unavailable."""


class PendingError(StateError):
    """The requested state has not reached a committed clean boundary."""
