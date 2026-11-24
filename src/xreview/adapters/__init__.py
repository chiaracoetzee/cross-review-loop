"""Built-in harness adapters."""

from xreview.adapters.agy import AgyAdapter
from xreview.adapters.base import (
    HarnessAdapter,
    TurnRequest,
    TurnResult,
    get_adapter,
    list_adapters,
    register_adapter,
)
from xreview.adapters.claude import ClaudeAdapter
from xreview.adapters.codex import CodexAdapter

register_adapter(CodexAdapter())
register_adapter(AgyAdapter())
register_adapter(ClaudeAdapter())

__all__ = [
    "AgyAdapter",
    "ClaudeAdapter",
    "CodexAdapter",
    "HarnessAdapter",
    "TurnRequest",
    "TurnResult",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
