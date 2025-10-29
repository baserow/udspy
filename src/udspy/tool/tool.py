"""Tool class for function wrapping and schema generation."""

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

from udspy.callback import with_callbacks
from udspy.utils.async_support import execute_function_async
from udspy.utils.schema import resolve_json_schema_reference


class Tool(BaseModel):
    """Wrapper for a tool function with metadata.

    Tools are callable functions that can be passed to Predict. The function
    signature and annotations are automatically converted to an OpenAI tool schema.
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
        """Initialize a Tool."""
        super().__init__(
            func=func,
            name=name or func.__name__,
            description=description or inspect.getdoc(func) or "",
            require_confirmation=require_confirmation,
        )
        self.args_schema = self.get_args_schema(resolve_defs=False)

    @property
    def _func(self) -> Callable[..., Any]:
        """Get the function wrapped with confirmation if required."""
        import functools

        from udspy.confirmation import check_tool_confirmation

        @functools.wraps(self.func)
        async def async_wrapper(**kwargs: Any) -> Any:
            if self.require_confirmation:
                kwargs = await check_tool_confirmation(self.name or "unknown", kwargs)
            return await execute_function_async(self.func, kwargs)

        return async_wrapper

    @with_callbacks
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function."""
        import asyncio

        coro = self._func(*args, **kwargs)
        try:
            asyncio.get_running_loop()
            return coro
        except RuntimeError:
            return asyncio.run(coro)

    @with_callbacks
    async def acall(self, **kwargs: Any) -> Any:
        """Async call the wrapped function."""
        return await self._func(**kwargs)

    @property
    def desc(self) -> str | None:
        """Alias for description (DSPy compatibility)."""
        return self.description

    @property
    def args(self) -> dict[str, Any] | None:
        """Alias for args_schema properties (DSPy compatibility)."""
        if self.args_schema and "properties" in self.args_schema:
            return self.args_schema["properties"]
        return self.args_schema

    def get_args_schema(self, resolve_defs: bool = True) -> dict[str, Any]:
        """Convert function arguments to a Pydantic model schema."""
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
        """Get output type name or schema."""
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
        """Convert to OpenAI tool schema."""
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


__all__ = ["Tool"]
