"""Adapter for formatting LLM inputs/outputs with Pydantic models."""

import enum
import inspect
import json
import re
from typing import Any, Literal, get_args, get_origin

from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from udspy.formatters import format_value, parse_value
from udspy.signature import Signature
from udspy.utils.schema import minimize_schema, resolve_json_schema_reference


def translate_field_type(field_name: str, field_info: FieldInfo) -> str:
    """Translate a field's type annotation into a format hint for the LLM.

    This function generates a placeholder with optional type constraints that guide
    the LLM on how to format output values for non-string types.

    Args:
        field_name: Name of the field
        field_info: Pydantic FieldInfo containing annotation

    Returns:
        Formatted string like "{field_name}" with optional type constraint comment

    Examples:
        For str: "{answer}"
        For int: "{count}\\n        # note: the value you produce must be a single int value"
        For bool: "{is_valid}\\n        # note: the value you produce must be True or False"
        For Literal: "{status}\\n        # note: the value you produce must exactly match one of: pending; approved"
    """
    field_type = field_info.annotation

    # For strings, no special formatting needed
    if field_type is str:
        desc = ""
    elif field_type is bool:
        desc = "must be True or False"
    elif field_type in (int, float):
        desc = f"must be a single {field_type.__name__} value"
    elif inspect.isclass(field_type) and issubclass(field_type, enum.Enum):
        enum_vals = "; ".join(str(member.value) for member in field_type)
        desc = f"must be one of: {enum_vals}"
    elif get_origin(field_type) is Literal:
        literal_values = get_args(field_type)
        desc = f"must exactly match (no extra characters) one of: {'; '.join([str(x) for x in literal_values])}"
    else:
        # For complex types (lists, dicts, Pydantic models), show JSON schema
        try:
            schema = minimize_schema(
                resolve_json_schema_reference(TypeAdapter(field_type).json_schema())
            )
            if schema.get("type") == "array":
                item_schema = schema.get("items", {}).get("properties", {})
                desc = f"must be a JSON array where every item adheres to the schema: {json.dumps(item_schema, ensure_ascii=False)}"
            else:
                desc = f"must adhere to the JSON schema: {json.dumps(schema, ensure_ascii=False)}"
        except Exception:
            # Fallback if we can't generate a schema
            desc = ""

    # Format with indentation for readability
    desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
    return f"{{{field_name}}}{desc}"


class ChatAdapter:
    """Adapter for formatting signatures into OpenAI chat messages.

    This adapter converts Signature inputs into properly formatted
    chat messages and parses LLM responses back into structured outputs.
    """

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Format example field structure with type hints for the LLM.

        Shows the LLM exactly how to structure inputs and outputs, including
        type constraints for non-string fields. This helps the LLM understand
        what format each field should use (e.g., integers, booleans, JSON objects).

        Args:
            signature: The signature defining input/output fields

        Returns:
            Formatted string showing field structure with type hints
        """
        parts = []
        parts.append(
            "All interactions will be structured in the following way, with the appropriate values filled in."
        )

        # Format input fields
        input_fields = signature.get_input_fields()
        if input_fields:
            for name, field_info in input_fields.items():
                type_hint = translate_field_type(name, field_info)
                parts.append(f"[[ ## {name} ## ]]\n{type_hint}")

        # Format output fields
        output_fields = signature.get_output_fields()
        if output_fields:
            for name, field_info in output_fields.items():
                type_hint = translate_field_type(name, field_info)
                parts.append(f"[[ ## {name} ## ]]\n{type_hint}")

        # Add completion marker
        parts.append("[[ ## completed ## ]]")

        return "\n\n".join(parts).strip()

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
            for i, (name, field_info) in enumerate(input_fields.items(), start=1):
                desc = field_info.description or ""
                parts.append(f"{i}. `{name}`: {desc}")

        # Add output field descriptions
        output_fields = signature.get_output_fields()
        if output_fields:
            parts.append("\n**Required Outputs:**")
            for i, (name, field_info) in enumerate(output_fields.items(), start=1):
                desc = field_info.description or ""
                parts.append(f"{i}. `{name}`: {desc}")

        input_field_names = ",".join([f"`{name}`" for name in input_fields.keys()])
        output_field_names = ",".join([f"`{name}`" for name in output_fields.keys()])
        parts.append(
            f"\nGiven the fields {input_field_names}, produce the fields {output_field_names}.\n"
        )

        parts.append(self.format_field_structure(signature))

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

        Uses regex to extract field content, ignoring any text before/after markers.
        Field content is stripped of leading/trailing whitespace (including newlines).

        Args:
            signature: The signature defining expected outputs
            completion: Raw completion string from LLM

        Returns:
            Dictionary of parsed output values
        """
        output_fields = signature.get_output_fields()
        outputs: dict[str, Any] = {}

        # Pattern: [[ ## field_name ## ]] followed by content until next marker or end
        # (?:...) = non-capturing group
        # [\s\S]*? = match any character (including newlines) non-greedily
        pattern = r"\[\[\s*##\s*(\w+)\s*##\s*\]\]\s*\n?([\s\S]*?)(?=\[\[\s*##\s*\w+\s*##\s*\]\]|$)"

        for match in re.finditer(pattern, completion):
            field_name = match.group(1).strip()
            content = match.group(
                2
            ).strip()  # strip() removes leading/trailing whitespace and newlines

            if field_name in output_fields:
                field_info = output_fields[field_name]

                # Parse according to field type
                try:
                    outputs[field_name] = parse_value(content, field_info.annotation)  # type: ignore[arg-type]
                except Exception:
                    # Fallback: keep as string
                    outputs[field_name] = content

        return outputs

    def format_tool_schema(self, tool: Any) -> dict[str, Any]:
        """Convert a Tool object or Pydantic model to OpenAI tool schema.

        This is where provider-specific schema formatting happens. The adapter
        takes the tool's normalized schema and converts it to OpenAI's expected format.

        Args:
            tool: Tool object or Pydantic model class

        Returns:
            OpenAI tool schema dictionary in the format:
            {
                "type": "function",
                "function": {
                    "name": str,
                    "description": str,
                    "parameters": dict  # Full JSON schema with type, properties, required
                }
            }
        """
        from udspy.tool import Tool

        if isinstance(tool, Tool):
            # Tool decorator - construct OpenAI schema from Tool properties
            # Tool.parameters gives us the complete resolved schema (type, properties, required)
            return {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,  # Already resolved, ready for OpenAI
                },
            }
        else:
            # Pydantic model - convert using existing logic
            tool_model = tool
            schema = tool_model.model_json_schema()

            # Extract description from docstring or schema
            description = (
                tool_model.__doc__.strip()
                if tool_model.__doc__
                else schema.get("description", f"Use {tool_model.__name__}")
            )

            # Build OpenAI function schema
            # Resolve any $defs references in the Pydantic schema
            tool_schema = resolve_json_schema_reference(
                {
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
            )

            return tool_schema

    def format_tool_schemas(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert Tool objects or Pydantic models to OpenAI tool schemas.

        Args:
            tools: List of Tool objects or Pydantic model classes

        Returns:
            List of OpenAI tool schema dictionaries
        """

        tool_schemas = []

        for tool_item in tools:
            tool_schema = self.format_tool_schema(tool_item)
            tool_schemas.append(tool_schema)

        return tool_schemas
