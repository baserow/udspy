"""Stream processing functions for Predict module."""

import asyncio
from typing import Any

import regex as re

from udspy.module.base import Module
from udspy.streaming import OutputStreamChunk, StreamEvent, ThoughtStreamChunk


def process_tool_call_delta(
    tool_calls: dict[int, dict[str, Any]],
    delta_tool_calls: list[Any],
) -> None:
    """Process and accumulate tool call deltas from streaming chunks.

    Updates the tool_calls dictionary in place with new delta information.

    Args:
        tool_calls: Dictionary to accumulate tool calls in (indexed by tool call index)
        delta_tool_calls: List of tool call deltas from the current chunk
    """
    for tool_call in delta_tool_calls:
        idx = tool_call.index
        if idx not in tool_calls:
            tool_calls[idx] = {
                "id": tool_call.id or "",
                "type": tool_call.type or "function",
                "function": {
                    "name": tool_call.function.name if tool_call.function else "",
                    "arguments": "",
                },
            }

        if tool_call.function and tool_call.function.arguments:
            tool_calls[idx]["function"]["arguments"] += tool_call.function.arguments


async def process_content_delta(
    module: Module,
    delta: str,
    acc_delta: str,
    current_field: str | None,
    accumulated_content: dict[str, list[str]],
    output_fields: dict[str, Any],
    field_pattern: re.Pattern[str],
    queue: asyncio.Queue[StreamEvent | None],
) -> tuple[str, str | None]:
    """Process content delta and emit field chunks to queue.

    Detects field boundaries using markers and emits streaming chunks for each field.
    Accumulates content until field boundaries are detected or pattern matching fails.

    Args:
        module: Module instance (for creating StreamEvent objects)
        delta: New content delta from current chunk
        acc_delta: Accumulated delta so far (not yet emitted)
        current_field: Name of current field being processed
        accumulated_content: Dictionary of accumulated content per field
        output_fields: Output fields from signature
        field_pattern: Regex pattern for detecting field markers
        queue: Event queue to emit chunks to

    Returns:
        Tuple of (updated acc_delta, updated current_field)
    """
    acc_delta += delta

    if not acc_delta:
        return acc_delta, current_field

    match = field_pattern.search(acc_delta)
    if match:
        if current_field:
            field_content = "".join(accumulated_content[current_field])
            await queue.put(
                OutputStreamChunk(module, current_field, "", field_content, is_complete=True)
            )

        current_field = match.group(1)
        acc_delta = match.group(2)

    if (
        current_field
        and current_field in output_fields
        and not field_pattern.match(acc_delta, partial=True)
    ):
        accumulated_content[current_field].append(acc_delta)
        field_content = "".join(accumulated_content[current_field])
        await queue.put(
            OutputStreamChunk(module, current_field, acc_delta, field_content, is_complete=False)
        )
        acc_delta = ""

    return acc_delta, current_field


async def process_reasoning_delta(
    module: Module,
    reasoning: ThoughtStreamChunk,
    choice: Any,
    queue: asyncio.Queue[StreamEvent | None],
) -> ThoughtStreamChunk:
    """Process reasoning delta from choice and emit thought chunks.

    Handles both native reasoning fields and AWS Bedrock's <reasoning> tags format.

    Args:
        module: Module instance (for creating StreamEvent objects)
        reasoning: Current reasoning chunk state
        choice: Choice object from streaming chunk
        queue: Event queue to emit chunks to

    Returns:
        Updated ThoughtStreamChunk
    """
    # For some reason, AWS Bedrock returns reasoning as content inside <reasoning> tags
    # instead of proper choice.delta.reasoning, so if that happens, we extract it here
    # Unfortunately, we cannot really say it's finished until we get the real content after.
    thought_chunk = choice.delta.content or ""
    if match := re.search(r"<reasoning>(.*?)</reasoning>", choice.delta.content or "", re.DOTALL):
        thought_chunk = match.group(1)
    else:
        thought_chunk = getattr(choice.delta, "reasoning", "")

    is_complete = choice.finish_reason is not None
    if thought_chunk or is_complete != reasoning.is_complete:
        reasoning = ThoughtStreamChunk(
            module,
            reasoning.field_name,
            thought_chunk,
            reasoning.content + thought_chunk,
            is_complete=is_complete,
        )
        await queue.put(reasoning)
    return reasoning


__all__ = ["process_tool_call_delta", "process_content_delta", "process_reasoning_delta"]
