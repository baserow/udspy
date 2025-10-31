"""Tool class for function wrapping and schema generation."""

import inspect
from collections.abc import Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, create_model

from udspy.callback import with_callbacks
from udspy.utils.async_support import execute_function_async
from udspy.utils.schema import resolve_json_schema_reference

# Type alias for JSON schema dictionaries
JsonSchema = dict[str, Any]


class Tool(BaseModel):
    """Wrapper for a tool function with metadata.

    Tools are callable functions that can be passed to Predict. The function
    signature and annotations are automatically converted to a JSON schema.

    The Tool class provides different schema representations for different use cases:
    - input_schema: Full JSON schema with resolved $defs (for internal use)
    - parameters: Same as input_schema, used for OpenAI function calling
    - format(): Human-readable description (for LLM prompts in modules like ReAct)

    Example:
        @tool(name="calculator", description="Perform arithmetic")
        def calculator(
            operation: str = Field(description="Operation type"),
            a: float = Field(description="First number"),
            b: float = Field(description="Second number"),
        ) -> float:
            ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
            return ops[operation]

        # For OpenAI function calling
        schema = calculator.parameters  # Full resolved schema

        # For LLM prompts in modules
        description = calculator.format()  # Human-readable string
    """

    func: Callable[..., Any]
    name: str | None = None
    description: str | None = None
    require_confirmation: bool = False

    # Internal: raw schema with $defs (cached at initialization)
    _raw_schema: JsonSchema | None = None

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
            require_confirmation: Whether to require user confirmation before execution
        """
        super().__init__(
            func=func,
            name=name or func.__name__,
            description=description or inspect.getdoc(func) or "",
            require_confirmation=require_confirmation,
        )
        # Generate and store the raw schema once at initialization
        self._raw_schema = self._generate_args_schema()

    @property
    def _func(self) -> Callable[..., Any]:
        """Get the function wrapped with argument parsing and confirmation if required."""
        import functools

        from udspy.confirmation import check_tool_confirmation

        @functools.wraps(self.func)
        async def async_wrapper(**kwargs: Any) -> Any:
            # Parse and validate arguments to convert JSON dicts to proper types
            parsed_kwargs = self.parse_and_validate_args(kwargs)

            if self.require_confirmation:
                parsed_kwargs = await check_tool_confirmation(self.name or "unknown", parsed_kwargs)
            return await execute_function_async(self.func, parsed_kwargs)

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
    def input_schema(self) -> JsonSchema:
        """Get the full JSON schema with all $defs resolved.

        This returns a complete JSON schema with all references resolved.
        Used internally for validation and processing.

        Returns:
            Fully resolved JSON schema dictionary with type, properties, and required fields
        """
        if self._raw_schema is None:
            return {}
        return resolve_json_schema_reference(self._raw_schema)

    @property
    def parameters(self) -> JsonSchema:
        """Get the parameters schema for OpenAI function calling.

        This is what OpenAI expects in the "parameters" field of a function schema.
        It includes type, properties, and required fields. Same as input_schema but
        with a clearer name for the OpenAI context.

        Returns:
            Complete JSON schema suitable for OpenAI function calling API
        """
        return self.input_schema

    @property
    def desc(self) -> str | None:
        """Alias for description (DSPy compatibility)."""
        return self.description

    @property
    def args_schema(self) -> JsonSchema:
        """Deprecated: Use .parameters instead.

        Returns the full resolved schema for backward compatibility.
        """
        return self.input_schema

    @property
    def args(self) -> JsonSchema | None:
        """Deprecated: Use .parameters['properties'] instead.

        Returns just the properties section for backward compatibility.
        """
        schema = self.input_schema
        if schema and "properties" in schema:
            return schema["properties"]
        return None

    def _generate_args_schema(self) -> JsonSchema:
        """Generate JSON schema from function signature.

        This is called once during initialization and the result is cached.
        It converts the function's type hints into a Pydantic model schema.

        Returns:
            Raw JSON schema (may contain $defs references)
        """
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)

        fields = {}
        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, str)
            default = ... if param.default == inspect.Parameter.empty else param.default
            fields[param_name] = (param_type, default)

        return create_model(f"{self.func.__name__}_args", **fields).model_json_schema()  # type: ignore[call-overload]

    def _get_args_model(self) -> type[BaseModel]:
        """Get or create the Pydantic model for tool arguments.

        This creates a Pydantic model based on the function's signature,
        which can be used to parse and validate arguments from JSON.

        Returns:
            Pydantic model class for the tool's arguments
        """
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)

        fields = {}
        for param_name, param in sig.parameters.items():
            param_type = type_hints.get(param_name, str)
            default = ... if param.default == inspect.Parameter.empty else param.default
            fields[param_name] = (param_type, default)

        return create_model(f"{self.func.__name__}_args", **fields)  # type: ignore[call-overload]

    def parse_and_validate_args(self, args: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate arguments from JSON dict to expected types.

        This method converts raw dict arguments (e.g., from JSON parsing) into
        the proper types expected by the tool function, including Pydantic models.

        Uses Pydantic validation to handle:
        - Type coercion for primitives (e.g., "123" -> 123 for int)
        - Parsing dicts into Pydantic model instances
        - Validation of all argument types and constraints

        Args:
            args: Raw arguments dict (e.g., from LLM JSON output)

        Returns:
            Parsed and validated arguments dict with proper types. Pydantic model
            parameters will be instances of those models, not dicts.

        Raises:
            ValidationError: If arguments don't match the expected schema

        Example:
            ```python
            @tool(name="create_row")
            def create_row(table_id: int, row: RowModel) -> str:
                ...

            # Raw args from LLM JSON
            raw_args = {"table_id": "123", "row": {"name": "test", "value": 42}}

            # Parse to proper types
            parsed_args = create_row.parse_and_validate_args(raw_args)
            # Result: {"table_id": 123, "row": RowModel(name="test", value=42)}
            ```
        """
        # Use Pydantic model to validate and coerce all arguments
        args_model = self._get_args_model()
        validated_model = args_model.model_validate(args)

        # Extract the validated values, keeping Pydantic models as instances
        sig = inspect.signature(self.func)
        parsed_args: dict[str, Any] = {}

        for param_name in sig.parameters.keys():
            if not hasattr(validated_model, param_name):
                continue

            value = getattr(validated_model, param_name)

            # Pydantic models are kept as instances, primitives are coerced
            parsed_args[param_name] = value

        return parsed_args

    def get_output_type_or_schema(self, resolve_defs: bool = True) -> str | JsonSchema:
        """Get output type name or schema.

        Args:
            resolve_defs: Whether to resolve $defs references (default: True)

        Returns:
            String type name for simple types, or dict schema for complex types
        """
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

    def format(self) -> str:
        """Format tool as human-readable string for LLM prompts.

        This creates a readable description of the tool including its name,
        description, and parameter schema. Used when tools are described in prompts.

        Returns:
            Human-readable tool description
        """
        desc = (self.description or "").replace("\n", " ").strip()
        desc_part = f", whose description is <desc>{desc}</desc>." if desc else "."

        # Get simplified parameter descriptions
        params = self.parameters.get("properties", {})
        arg_desc = f"It takes arguments {params}." if params else "It takes no arguments."

        return f"{self.name}{desc_part} {arg_desc}"

    def __str__(self) -> str:
        """String representation of the tool."""
        return self.format()


__all__ = ["Tool"]
