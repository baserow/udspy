"""Utility functions for udspy."""

import asyncio
import inspect
from collections.abc import Callable
from typing import Any


def ensure_sync_context(method_name: str) -> None:
    """Ensure we're NOT in an async context (for sync methods).

    This prevents calling sync methods like forward() or resume() from within
    async code, which would fail because asyncio.run() can't be called when
    there's already a running event loop.

    Args:
        method_name: Name of the method being called (for error message)

    Raises:
        RuntimeError: If called from within an async context, with a helpful
            message suggesting the async alternative

    Example:
        ```python
        def forward(self, **inputs):
            ensure_sync_context("forward")  # Check first
            return asyncio.run(self.aforward(**inputs))
        ```

    Note:
        This uses asyncio.get_running_loop() which raises RuntimeError
        if there's no event loop. We catch that specific error to allow
        the sync method to proceed.
    """
    try:
        asyncio.get_running_loop()
        # If we get here, there IS a running loop - we're in async context
        # Extract class name from method_name if it's in "ClassName.method" format
        if "." in method_name:
            class_name, method = method_name.rsplit(".", 1)
            async_method = f"await {class_name[0].lower() + class_name[1:]}.a{method}(...)"
        else:
            async_method = f"await ...a{method_name}(...)"

        raise RuntimeError(
            f"Cannot call {method_name}() from async context. Use '{async_method}' instead."
        )
    except RuntimeError as e:
        # Check if it's the "no running event loop" error (which is what we want)
        if "no running event loop" not in str(e).lower():
            # It's a different RuntimeError - re-raise it
            raise


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


def format_tool_exception(err: BaseException, *, limit: int = 5) -> str:
    """
    Return a one-string traceback summary.
    * `limit` - how many stack frames to keep (from the innermost outwards).
    """

    import traceback

    return (
        "\n"
        + "".join(
            traceback.format_exception(type(err), err, err.__traceback__, limit=limit)
        ).strip()
    )


async def execute_function_async(func: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    """Execute a function asynchronously, handling both sync and async functions.

    Args:
        func: Function to execute (can be sync or async)
        kwargs: Keyword arguments to pass

    Returns:
        Function result
    """
    if inspect.iscoroutinefunction(func):
        return await func(**kwargs)
    else:
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(**kwargs))


__all__ = [
    "ensure_sync_context",
    "resolve_json_schema_reference",
    "minimize_schema",
    "format_tool_exception",
    "execute_function_async",
]
