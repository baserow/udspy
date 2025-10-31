"""Helper types for tool management."""

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from udspy.utils.schema import minimize_schema

if TYPE_CHECKING:
    from udspy.tool.tool import Tool


class Tools(BaseModel):
    """Container for multiple Tool instances."""

    tools: list["Tool"]

    def format(self, include_output_type: bool = False) -> str:
        """Format all tools as a string."""
        parts = []
        defs: dict[str, Any] = {}
        for idx, tool in enumerate(self.tools, start=1):
            # Get raw schema with $defs (if available)
            if tool._raw_schema:
                tool_args_schema = tool._raw_schema.copy()
                tool_defs = tool_args_schema.pop("$defs", {})
                defs.update({k: minimize_schema(v)["properties"] for k, v in tool_defs.items()})
                tool_args_schema = minimize_schema(tool_args_schema.get("properties", {}))
            else:
                tool_args_schema = {}

            # Format tool description
            desc = (tool.description or "").replace("\n", " ").strip()
            desc_part = f", whose description is <desc>{desc}</desc>." if desc else "."
            arg_desc = (
                f"It takes arguments {json.dumps(tool_args_schema)}."
                if tool_args_schema
                else "It takes no arguments."
            )
            fmt_tool = f"({idx}): {tool.name}{desc_part} {arg_desc}"

            if include_output_type:
                output_type = tool.get_output_type_or_schema(resolve_defs=False)
                if isinstance(output_type, dict):
                    defs.update(output_type.pop("$defs", {}))
                    output_type = minimize_schema(output_type["properties"])
                fmt_tool += f" It returns {output_type}."

            parts.append(fmt_tool)

        if defs:
            parts.append(f"\n## Tool Definitions ($defs):\n {json.dumps(defs)}\n")
        return "\n".join(parts)


class ToolCall(BaseModel):
    """Container for a single tool call."""

    call_id: str | None = None
    name: str
    args: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for backward compatibility."""
        if key == "id":
            return self.call_id
        elif key == "arguments":
            return json.dumps(self.args)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like setting for backward compatibility."""
        if key == "id":
            self.call_id = value
        elif key == "arguments":
            self.args = value
        else:
            setattr(self, key, value)


class ToolCalls(BaseModel):
    """Container for multiple tool calls."""

    tool_calls: list[ToolCall]


__all__ = ["Tools", "ToolCall", "ToolCalls"]
