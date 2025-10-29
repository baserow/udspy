"""Tool decorator for native function calling."""

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

from udspy.callback import with_callbacks
from udspy.utils import execute_function_async, minimize_schema, resolve_json_schema_reference


class Tool(BaseModel):
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

    func: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    args_schema: dict[str, Any] | None = None
    require_confirmation: bool = False

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        require_confirmation: bool = False,
    ):
        """Initialize a Tool.

        Args:
            func: The function to wrap
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            require_confirmation: If True, wraps function with @confirm_first decorator
            callbacks: Optional list of callback handlers for this tool instance
        """
        super().__init__(
            func=func,
            name=name or func.__name__,
            description=description or inspect.getdoc(func) or "",
            require_confirmation=require_confirmation,
        )
        self.args_schema = self.get_args_schema(resolve_defs=False)

    @property
    def _func(self) -> Callable[..., Any]:
        """Get the function wrapped with confirmation if required.

        Returns a coroutine function that can be awaited.
        """
        import functools

        from udspy.confirmation import check_tool_confirmation

        @functools.wraps(self.func)
        async def async_wrapper(**kwargs: Any) -> Any:
            # Check confirmation if required
            if self.require_confirmation:
                kwargs = await check_tool_confirmation(self.name or "unknown", kwargs)

            # Execute the function
            return await execute_function_async(self.func, kwargs)

        return async_wrapper

    @with_callbacks
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function.

        If called from an async context, returns a coroutine that can be awaited.
        If called from a sync context, runs the coroutine using asyncio.run().
        """
        import asyncio

        coro = self._func(*args, **kwargs)

        # Check if we're in an async context by trying to get the running loop
        try:
            asyncio.get_running_loop()
            # We're in an async context - return the coroutine to be awaited
            return coro
        except RuntimeError:
            # Not in an async context - run it synchronously
            return asyncio.run(coro)

    @with_callbacks
    async def acall(self, **kwargs: Any) -> Any:
        """Async call the wrapped function.

        If the function is async, awaits it. Otherwise, runs it in executor.
        If require_confirmation is True, handles confirmation before execution.

        Args:
            **kwargs: Arguments to pass to the function

        Returns:
            Function result

        Raises:
            ConfirmationRequired: If require_confirmation is True and not approved
            ConfirmationRejected: If user rejected the operation
        """
        return await self._func(**kwargs)

    @property
    def desc(self) -> str | None:
        """Alias for description (DSPy compatibility)."""
        return self.description

    @property
    def args(self) -> dict[str, Any] | None:
        """Alias for args_schema properties (DSPy compatibility).

        Returns just the properties dict for easier field access.
        """
        if self.args_schema and "properties" in self.args_schema:
            return self.args_schema["properties"]
        return self.args_schema

    def get_args_schema(self, resolve_defs: bool = True) -> dict[str, Any]:
        """Convert function arguments to a Pydantic model.

        Returns:
            Pydantic model class representing the function's arguments
        """
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)

        fields = {}
        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, str)
            default = ... if param.default == inspect.Parameter.empty else param.default
            fields[param_name] = (param_type, default)

        schema = create_model(f"{self.func.__name__}_args", **fields).model_json_schema()  # type: ignore[call-overload]
        if resolve_defs:
            return resolve_json_schema_reference(schema)["properties"]
        else:
            return schema

    def get_output_type_or_schema(self, resolve_defs: bool = True) -> str | dict[str, Any]:
        """If it's a native type, return the type name. If it's a Pydantic model, return its schema."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            type(None): "null",
        }
        return_type = get_type_hints(self.func).get("return", None)
        if return_type is not None and hasattr(return_type, "model_json_schema"):
            schema = return_type.model_json_schema()
            if resolve_defs:
                return resolve_json_schema_reference(schema)["properties"]
            else:
                return schema
        elif (valid_type := type_map.get(return_type)) is not None:  # type: ignore[arg-type]
            return valid_type
        else:
            raise ValueError(
                f"Unsupported return type for tool: {return_type}. "
                "It must either be a native type (str, int, float, bool) or a Pydantic model."
            )

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI tool schema.

        Returns:
            OpenAI-compatible tool schema dictionary
        """

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": resolve_json_schema_reference(
                    self.get_args_schema(resolve_defs=False)
                ),
            },
        }

    def format(self, name: str, description: str, args_schema: dict[str, Any]) -> str:
        desc = description.replace("\n", " ").strip()
        desc = f", whose description is <desc>{desc}</desc>." if desc else "."
        arg_desc = f"It takes arguments {args_schema}."
        return f"{name}{desc} {arg_desc}"

    def __str__(self) -> str:
        if self.args_schema:
            args_schema = resolve_json_schema_reference(self.args_schema)
        return self.format(self.name or "unnamed_tool", self.description or "", args_schema)


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

        # Tool that requires confirmation (e.g., destructive operations)
        @tool(name="DeleteFile", description="Delete a file", require_confirmation=True)
        def delete_file(path: str = Field(...)) -> str:
            os.remove(path)
            return f"Deleted {path}"
        ```
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        return Tool(
            func,
            name=name,
            description=description,
            require_confirmation=require_confirmation,
        )

    return decorator


class Tools(BaseModel):
    """Container for multiple Tool instances."""

    tools: list[Tool]

    def format(self, include_output_type: bool = False) -> str:
        """Format all tools as a string."""

        parts = []
        defs: dict[str, Any] = {}
        for idx, tool in enumerate(self.tools, start=1):
            if tool.args_schema:
                tool_args_schema = tool.get_args_schema(resolve_defs=False)
                defs.update(tool_args_schema.pop("$defs", {}))
                tool_args_schema = minimize_schema(tool_args_schema["properties"])
            else:
                tool_args_schema = {}
            fmt_tool = f"({idx}): {tool.format(tool.name or 'unknown', tool.description or '', tool_args_schema)}"

            if include_output_type:
                output_type = tool.get_output_type_or_schema(resolve_defs=False)
                if isinstance(output_type, dict):
                    defs.update(output_type.pop("$defs", {}))
                    output_type = minimize_schema(output_type["properties"])
                fmt_tool += f" It returns {output_type}."

            parts.append(fmt_tool)

        # Prepend common definitions if any
        if defs:
            parts.insert(0, f"Common tools definitions: {defs}\n")
        return "\n".join(parts)


class ToolCall(BaseModel):
    """Container for a single tool call."""

    call_id: str | None = None
    name: str
    args: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access for backward compatibility."""
        import json

        if key == "id":
            return self.call_id
        elif key == "arguments":
            # Return JSON string representation of args for compatibility
            return json.dumps(self.args)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like setting for backward compatibility."""
        if key == "id":
            self.call_id = value
        elif key == "arguments":
            # Map 'arguments' to 'args' for compatibility
            self.args = value
        else:
            setattr(self, key, value)


class ToolCalls(BaseModel):
    """Container for multiple tool calls."""

    tool_calls: list["ToolCall"]
