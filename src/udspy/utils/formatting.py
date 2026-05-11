"""Formatting utility functions."""


def format_tool_exception(err: BaseException, *, limit: int = 3, max_length: int = 1000) -> str:
    """
    Return a concise error summary with the error message first (most useful
    for the LLM), followed by abbreviated stack frames if space allows.

    * `limit` - how many stack frames to keep (from the innermost outwards).
    * `max_length` - maximum length of the returned string.
    """

    import traceback

    # Error message first — this is what the LLM needs most
    error_line = f"{type(err).__name__}: {err}"

    # Then abbreviated stack frames for context
    tb_lines = traceback.format_tb(err.__traceback__, limit=limit)
    frames = "".join(tb_lines).strip()

    formatted = f"{error_line}\n{frames}" if frames else error_line

    if len(formatted) > max_length:
        formatted = formatted[:max_length] + "..."

    return formatted


def format_validation_error(tool_name: str, err: BaseException, tool=None) -> str:
    """
    Format a validation/type error with the expected schema appended,
    so the LLM knows how to fix its arguments.
    """

    parts = [f"Validation error for '{tool_name}': {err}."]
    if tool is not None:
        parts.append(f"Expected tool args schema: {tool.parameters}.")
    return " ".join(parts)


__all__ = [
    "format_tool_exception",
    "format_validation_error",
]
