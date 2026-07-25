"""Small compatibility helpers shared by Qwen3-ASR parity tools."""

from __future__ import annotations

import sys
import types


def enable_local_qwen_asr() -> None:
    """Adapt the local Qwen3-ASR checkout to the installed Transformers 4.56."""
    try:
        import nagisa  # noqa: F401
    except ImportError:
        sys.modules["nagisa"] = types.ModuleType("nagisa")

    import transformers.utils as utils
    import transformers.utils.generic as generic

    original = generic.check_model_inputs

    def compatible(func=None):
        return original if func is None else original(func)

    generic.check_model_inputs = compatible
    utils.check_model_inputs = compatible
