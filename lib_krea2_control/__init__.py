"""Krea2 Control internals: model forward (forward.py) and image helpers (imaging.py)."""


class ControlError(RuntimeError):
    """A problem the user can fix (no control image, unknown LoRA...). Reported as one clear line, without a traceback."""
