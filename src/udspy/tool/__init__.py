"""Tool system for native function calling."""

from udspy.tool.decorator import tool
from udspy.tool.tool import Tool
from udspy.tool.types import ToolCall, ToolCalls, Tools

__all__ = [
    "Tool",
    "tool",
    "Tools",
    "ToolCall",
    "ToolCalls",
]
