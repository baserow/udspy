"""JSON schema utility functions."""

from typing import Any


def resolve_json_schema_reference(schema: dict) -> dict:
    """Recursively resolve json model schema, expanding all references."""

    # If there are no definitions to resolve, return the main schema
    if "$defs" not in schema and "definitions" not in schema:
        return schema

    def resolve_refs(obj: Any) -> Any:
        if not isinstance(obj, (dict, list)):
            return obj
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_path = obj["$ref"].split("/")[-1]
                return resolve_refs(schema["$defs"][ref_path])
            return {k: resolve_refs(v) for k, v in obj.items()}

        # Must be a list
        return [resolve_refs(item) for item in obj]

    # Resolve all references in the main schema
    resolved_schema = resolve_refs(schema)
    # Remove the $defs key as it's no longer needed
    resolved_schema.pop("$defs", None)
    return resolved_schema


def minimize_schema(schema: dict[str, Any], keep_description: bool = True) -> dict[str, Any]:
    """
    Remove unnecessary fields from JSON schema.

    Args:
        schema: The JSON schema dict
        keep_description: Whether to keep description fields (useful for LLMs)
    """
    fields_to_remove = [
        "title",
        "default",  # Remove if you don't need defaults in schema
        "examples",
        "additionalProperties",
        "$defs",  # Remove $defs if you inline everything
        "definitions",
    ]

    if not keep_description:
        fields_to_remove.append("description")

    def clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            # Remove unwanted fields
            for field in fields_to_remove:
                obj.pop(field, None)

            # Recursively clean nested objects
            for _key, value in list(obj.items()):
                if isinstance(value, (dict, list)):
                    clean(value)

        elif isinstance(obj, list):
            for item in obj:
                clean(item)

        return obj

    return clean(schema)


__all__ = [
    "resolve_json_schema_reference",
    "minimize_schema",
]
