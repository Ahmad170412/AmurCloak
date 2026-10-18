"""AmurCloak engine - local face detection and redaction.

Import order note: ``presets`` is imported first everywhere because it is the
single source of truth for tuning values. Nothing in this package reads a
default from anywhere else.
"""

__all__ = ["detector", "photo", "pipeline", "presets", "redact", "sessions"]
