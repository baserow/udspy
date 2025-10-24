"""Tool decorator for native function calling."""

import inspect
from collections.abc import Callable
from typing import Any, get_args, get_origin

from pydantic.fields import FieldInfo


class Tool:
    """Wrapper for a tool function with metadata.

    Tools are callable functions that can be passed to Predict. The function
    signature and annotations are automatically converted to an OpenAI tool schema.

    Example:
        ```python
        @tool(name="Calculator", description="Perform arithmetic operations")
        def calculator(
            operation: str = Field(description="add, subtract, multiply, divide"),
            a: float = Field(description="First number"),
            b: float = Field(description="Second number"),
        ) -> float:
            ops = {"add": a + b, "multiply": a * b, ...}
            return ops[operation]

        predictor = Predict(QA, tools=[calculator])
        ```
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ):
        """Initialize a Tool.

        Args:
            func: The function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
        """
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or ""

        # Extract parameter schema from function signature
        sig = inspect.signature(func)
        self.parameters: dict[str, dict[str, Any]] = {}

        for param_name, param in sig.parameters.items():
            # Skip *args, **kwargs
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            param_info: dict[str, Any] = {
                "type": param.annotation if param.annotation != inspect.Parameter.empty else str,
                "description": None,
                "required": param.default == inspect.Parameter.empty,
            }

            # Extract Field metadata if present
            if isinstance(param.default, FieldInfo):
                param_info["description"] = param.default.description
                param_info["required"] = param.default.is_required()

            self.parameters[param_name] = param_info

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function."""
        return self.func(*args, **kwargs)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI tool schema.

        Returns:
            OpenAI-compatible tool schema dictionary
        """
        properties = {}
        required = []

        for name, info in self.parameters.items():
            prop: dict[str, Any] = {"type": self._python_type_to_json_type(info["type"])}

            if info["description"]:
                prop["description"] = info["description"]

            properties[name] = prop

            if info["required"]:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _python_type_to_json_type(python_type: Any) -> str:
        """Convert Python type annotation to JSON Schema type.

        Args:
            python_type: Python type annotation

        Returns:
            JSON Schema type string
        """
        # Handle Optional[T] -> T | None
        origin = get_origin(python_type)
        if origin is not None:
            args = get_args(python_type)
            # For Union types (including Optional), try to get the non-None type
            if len(args) > 0:
                for arg in args:
                    if arg is not type(None):  # noqa: E721
                        python_type = arg
                        break

        # Map Python types to JSON Schema types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        return type_map.get(python_type, "string")


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator to mark a function as a tool.

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)

    Returns:
        Decorator function

    Example:
        ```python
        @tool(name="Calculator", description="Perform arithmetic operations")
        def calculator(
            operation: str = Field(description="add, subtract, multiply, divide"),
            a: float = Field(description="First number"),
            b: float = Field(description="Second number"),
        ) -> float:
            ops = {
                "add": a + b,
                "subtract": a - b,
                "multiply": a * b,
                "divide": a / b if b != 0 else float("inf"),
            }
            return ops[operation]
        ```
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(func, name=name, description=description)

    return decorator
