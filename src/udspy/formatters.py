"""Value formatting and parsing utilities for LLM inputs/outputs."""

import json
from typing import Any

from pydantic import BaseModel


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


__all__ = [
    "format_value",
    "parse_value",
]
