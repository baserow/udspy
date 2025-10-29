"""Tool decorator function."""

from collections.abc import Callable
from typing import Any

from udspy.tool.tool import Tool


def tool(
    name: str | None = None,
    description: str | None = None,
    *,
    require_confirmation: bool = False,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator to mark a function as a tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        require_confirmation: If True, wraps function with @confirm_first decorator

    Returns:
        Decorator function
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(
            func,
            name=name,
            description=description,
            require_confirmation=require_confirmation,
        )

    return decorator


__all__ = ["tool"]
