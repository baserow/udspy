"""Adapter for formatting LLM inputs/outputs with Pydantic models."""

import json
from typing import Any

from pydantic import BaseModel

from udspy.signature import Signature


def format_value(value: Any) -> str:
    """Format a value for inclusion in a prompt.

    Args:
        value: The value to format

    Returns:
        Formatted string representation
    """
    if isinstance(value, str):
        return value
    elif isinstance(value, (list, dict)):
        return json.dumps(value, indent=2)
    elif isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    else:
        return str(value)


def parse_value(value_str: str, type_: type) -> Any:
    """Parse a string value into the specified type.

    Args:
        value_str: String value to parse
        type_: Target type

    Returns:
        Parsed value
    """
    # Handle strings
    if type_ is str:
        return value_str.strip()

    # Handle numeric types
    if type_ is int:
        return int(value_str.strip())
    if type_ is float:
        return float(value_str.strip())
    if type_ is bool:
        return value_str.strip().lower() in ("true", "yes", "1")

    # Handle Pydantic models
    try:
        if isinstance(type_, type) and issubclass(type_, BaseModel):
            # Try parsing as JSON first
            try:
                data = json.loads(value_str)
                return type_.model_validate(data)
            except json.JSONDecodeError:
                # Fallback: treat as JSON string
                return type_.model_validate_json(value_str)
    except (TypeError, ValueError):
        pass

    # Handle lists and dicts
    try:
        parsed = json.loads(value_str)
        if isinstance(parsed, (list, dict)):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: return as string
    return value_str.strip()


class ChatAdapter:
    """Adapter for formatting signatures into OpenAI chat messages.

    This adapter converts Signature inputs into properly formatted
    chat messages and parses LLM responses back into structured outputs.
    """

    def format_instructions(self, signature: type[Signature]) -> str:
        """Format signature instructions and field descriptions.

        Args:
            signature: The signature to format

        Returns:
            Formatted instruction string
        """
        parts = []

        # Add main instructions
        instructions = signature.get_instructions()
        if instructions:
            parts.append(instructions)

        # Add input field descriptions
        input_fields = signature.get_input_fields()
        if input_fields:
            parts.append("\n**Inputs:**")
            for name, field_info in input_fields.items():
                desc = field_info.description or ""
                parts.append(f"- `{name}`: {desc}")

        # Add output field descriptions
        output_fields = signature.get_output_fields()
        if output_fields:
            parts.append("\n**Required Outputs:**")
            for name, field_info in output_fields.items():
                desc = field_info.description or ""
                parts.append(f"- `{name}`: {desc}")

        # Add output format instructions
        parts.append("\n**Output Format:**\nStructure your response with clear field markers:\n")
        for name in output_fields:
            parts.append(f"[[ ## {name} ## ]]\n<your {name} here>")

        return "\n".join(parts)

    def format_inputs(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> str:
        """Format input values into a message.

        Args:
            signature: The signature defining expected inputs
            inputs: Dictionary of input values

        Returns:
            Formatted input string
        """
        parts = []
        input_fields = signature.get_input_fields()

        for name, _field_info in input_fields.items():
            if name in inputs:
                value = inputs[name]
                formatted = format_value(value)
                parts.append(f"[[ ## {name} ## ]]\n{formatted}")

        return "\n\n".join(parts)

    def parse_outputs(
        self,
        signature: type[Signature],
        completion: str,
    ) -> dict[str, Any]:
        """Parse LLM completion into structured outputs.

        Args:
            signature: The signature defining expected outputs
            completion: Raw completion string from LLM

        Returns:
            Dictionary of parsed output values
        """
        output_fields = signature.get_output_fields()
        outputs: dict[str, Any] = {}

        # Split completion into sections by field markers
        sections: list[tuple[str | None, list[str]]] = [(None, [])]

        for line in completion.splitlines():
            # Check for field marker: [[ ## field_name ## ]]
            if line.strip().startswith("[[ ## ") and line.strip().endswith(" ## ]]"):
                field_name = line.strip()[6:-5].strip()
                sections.append((field_name, []))
            else:
                sections[-1][1].append(line)

        # Parse each section
        for field_name, lines in sections:  # type: ignore[assignment]
            if field_name and field_name in output_fields:
                field_info = output_fields[field_name]
                value_str = "\n".join(lines).strip()

                # Parse according to field type
                try:
                    outputs[field_name] = parse_value(value_str, field_info.annotation)  # type: ignore[arg-type]
                except Exception:
                    # Fallback: keep as string
                    outputs[field_name] = value_str

        return outputs

    def format_tool_schemas(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert Tool objects or Pydantic models to OpenAI tool schemas.

        Args:
            tools: List of Tool objects or Pydantic model classes

        Returns:
            List of OpenAI tool schema dictionaries
        """
        from udspy.tool import Tool

        tool_schemas = []

        for tool_item in tools:
            if isinstance(tool_item, Tool):
                # Tool decorator - use its built-in schema conversion
                tool_schemas.append(tool_item.to_openai_schema())
            else:
                # Pydantic model - convert using existing logic
                tool_model = tool_item
                schema = tool_model.model_json_schema()

                # Extract description from docstring or schema
                description = (
                    tool_model.__doc__.strip()
                    if tool_model.__doc__
                    else schema.get("description", f"Use {tool_model.__name__}")
                )

                # Build OpenAI function schema
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": schema.get("title", tool_model.__name__),
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": schema.get("properties", {}),
                            "required": schema.get("required", []),
                            "additionalProperties": False,
                        },
                    },
                }

                # Remove $defs if present (internal Pydantic references)
                if "$defs" in tool_schema["function"]["parameters"]:  # type: ignore[index]
                    del tool_schema["function"]["parameters"]["$defs"]  # type: ignore[index]

                tool_schemas.append(tool_schema)

        return tool_schemas
