"""Tool execution functions for Predict module."""

import json

from openai import BaseModel

from udspy.history import History
from udspy.tool import Tool, ToolCall


async def execute_tool_calls(
    tools: dict[str, Tool],
    native_tool_calls: list[ToolCall],
    history: History,
) -> None:
    """Execute tool calls and add results to history.

    First adds the assistant message with tool calls, then executes each tool
    and adds the results as tool messages to history.

    Args:
        tools: Dictionary mapping tool names to Tool instances
        native_tool_calls: List of tool calls to execute
        history: History object to update with tool calls and results
    """
    # Add assistant message with tool calls
    history.add_assistant_message(
        tool_calls=[
            {
                "id": tc.call_id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
            }
            for tc in native_tool_calls
        ]
    )

    # Execute each tool and add results
    for tool_call in native_tool_calls:
        call_id = tool_call.call_id
        tool_name = tool_call.name
        tool_args = tool_call.args

        content: str = ""
        if tool_name in tools:
            try:
                content = await tools[tool_name](**tool_args)
                if isinstance(content, BaseModel):
                    content = content.model_dump_json()
                elif not isinstance(content, str):
                    content = json.dumps(content)
            except Exception as e:
                content = f"Error executing tool: {e}"
        else:
            content = f"Error: Tool {tool_name} not found"

        history.add_tool_result(str(call_id), content)


__all__ = ["execute_tool_calls"]
