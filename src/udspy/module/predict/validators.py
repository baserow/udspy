"""Validation functions for Predict module."""

from typing import Any

from udspy.exceptions import AdapterParseError
from udspy.signature import Signature
from udspy.tool import Tool, ToolCall


def validate_inputs(signature: type[Signature], inputs: dict[str, Any]) -> None:
    """Validate that all required inputs are provided.

    Args:
        signature: Signature class defining required inputs
        inputs: Input values provided by user

    Raises:
        ValueError: If a required input field is missing
    """
    input_fields = signature.get_input_fields()
    for field_name in input_fields:
        if field_name not in inputs:
            raise ValueError(f"Missing required input field: {field_name}")


def check_valid_outputs_or_raise(
    adapter_name: str,
    signature: type[Signature],
    tools: dict[str, Tool] | None,
    native_tool_calls: list[ToolCall],
    outputs: dict[str, Any],
    completion_text: str,
) -> None:
    """Check if tool calls and outputs are valid; raise AdapterParseError if not.

    Args:
        adapter_name: Name of the adapter (for error messages)
        signature: Signature class defining expected outputs
        tools: Dictionary of available tools (if any)
        native_tool_calls: Tool calls from LLM
        outputs: Parsed outputs from LLM
        completion_text: Raw completion text from LLM

    Raises:
        AdapterParseError: If tool calls reference unknown tools or outputs don't match signature
    """
    # Verify tool calls refer to known tools (only if we have tools configured)
    if tools:
        for tool_call in native_tool_calls:
            tool_name = tool_call.name
            if tool_name and tool_name not in tools:
                raise AdapterParseError(
                    adapter_name=adapter_name,
                    signature=signature,
                    lm_response="",
                    parsed_result={"error": f"Tool '{tool_name}' not found among available tools."},
                )

    # If no tool calls, verify outputs match signature fields
    if not native_tool_calls and outputs.keys() != signature.get_output_fields().keys():
        raise AdapterParseError(
            adapter_name=adapter_name,
            signature=signature,
            lm_response=completion_text,
            parsed_result=outputs,
        )


__all__ = ["validate_inputs", "check_valid_outputs_or_raise"]
