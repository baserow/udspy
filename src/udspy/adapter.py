"""Adapter for formatting LLM inputs/outputs with Pydantic models."""

import enum
import inspect
import json
from typing import Any, Literal, get_args, get_origin

import jiter
import regex as re
from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo

from udspy.exceptions import AdapterParseError
from udspy.formatters import format_value, parse_value
from udspy.lm import ChatCompletion, ChatCompletionChunk
from udspy.signature import Signature
from udspy.streaming import OutputStreamChunk, ThoughtStreamChunk, emit_event
from udspy.tool import Tool, ToolCall
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


class StreamingParser:
    """Parse streaming responses and generate StreamEvent objects.

    This parser processes streaming chunks from the LLM, handling:
    - Content deltas (JSON output fields)
    - Tool call deltas
    - Reasoning deltas

    It yields StreamEvent objects as they occur and provides finalized
    outputs at the end.
    """

    def __init__(
        self,
        adapter: "ChatAdapter",
        module: Any,
        signature: Any,
    ):
        """Initialize streaming parser.

        Args:
            adapter: ChatAdapter instance for parsing logic
            module: Module instance for creating stream events
            signature: Signature defining expected outputs
        """
        self.adapter = adapter
        self.module = module
        self.signature = signature
        self.output_fields = signature.get_output_fields()

        # Content parsing state
        self.accumulated_json = ""
        self.previous_values: dict[str, str] = {}
        self.completed_fields: set[str] = set()

        # Tool call accumulation
        self.tool_calls: dict[int, dict[str, Any]] = {}

        # Reasoning tracking
        self.reasoning_content = ""
        self.reasoning_complete = False
        self.has_seen_reasoning = False  # Track if we've seen any reasoning

        # Full completion text
        self.full_completion: list[str] = []

    def reset_content_accumulator(self) -> None:
        """Reset the JSON content accumulator.

        Called when reasoning is completed and we're about to parse actual output fields.
        """
        self.accumulated_json = ""
        self.previous_values.clear()
        self.completed_fields.clear()

    async def process_chunk(self, chunk: ChatCompletionChunk) -> Any:
        """Process a streaming chunk and yield StreamEvent objects.

        Args:
            chunk: ChatCompletionChunk from the LLM

        Yields:
            StreamEvent objects (ThoughtStreamChunk, OutputStreamChunk, etc.)
        """

        choice = chunk.choices[0]

        # Process tool calls first (mutually exclusive with content/reasoning)
        if choice.delta.tool_calls:
            self.adapter.process_tool_call_deltas(self.tool_calls, choice.delta.tool_calls)
            return

        # Check if the answer contains reasoning content
        (
            reasoning_delta,
            remaining_delta,
        ) = self.adapter.split_reasoning_and_content_delta(chunk)
        # If we found reasoning content, track that we're in reasoning mode
        if reasoning_delta and not self.reasoning_complete:
            self.has_seen_reasoning = True
            self.reasoning_content += reasoning_delta
            self.reasoning_complete = bool(remaining_delta)

            yield ThoughtStreamChunk(
                self.module,
                "thought",
                reasoning_delta,
                self.reasoning_content,
                is_complete=self.reasoning_complete,
            )
            return

        # Process content delta for output fields
        # Only accumulate if: no reasoning seen OR reasoning is complete
        if remaining_delta:
            self.full_completion.append(remaining_delta)
            async for event in self._process_content_delta(remaining_delta):
                yield event

    async def _process_content_delta(self, delta: str) -> Any:
        """Process content delta for JSON output fields.

        Args:
            delta: New content fragment

        Yields:
            OutputStreamChunk events
        """

        if not delta:
            return

        self.accumulated_json += delta

        # Try to parse the accumulated JSON
        try:
            parsed = jiter.from_json(
                self.accumulated_json.encode("utf-8"), partial_mode="trailing-strings"
            )
        except (TypeError, ValueError):
            # If we can't parse yet, just accumulate more
            return

        if not isinstance(parsed, dict):
            return

        # Process each field in the parsed output
        for field_name, value in parsed.items():
            if field_name not in self.output_fields:
                continue

            value_str = str(value) if not isinstance(value, str) else value
            previous = self.previous_values.get(field_name, "")

            if value_str != previous:
                delta_content = value_str[len(previous) :]
                yield OutputStreamChunk(
                    module=self.module,
                    field_name=field_name,
                    delta=delta_content,
                    content=value_str,
                    is_complete=False,
                )
                self.previous_values[field_name] = value_str
            elif value_str and field_name not in self.completed_fields:
                yield OutputStreamChunk(
                    module=self.module,
                    field_name=field_name,
                    delta="",
                    content=value_str,
                    is_complete=True,
                )
                self.completed_fields.add(field_name)

    async def finalize(self) -> tuple[dict[str, Any], list[Any], str]:
        """Finalize parsing and return outputs, tool calls, and completion text.

        Returns:
            Tuple of (outputs, native_tool_calls, completion_text)

        Raises:
            AdapterParseError: If parsing fails
        """
        # Parse final outputs
        outputs = self.adapter.parse_outputs(self.signature, self.accumulated_json)

        # Emit completion events for any fields not yet marked complete
        for field_name in self.output_fields:
            if field_name in outputs and field_name not in self.completed_fields:
                value = outputs[field_name]
                value_str = str(value) if not isinstance(value, str) else value
                emit_event(
                    OutputStreamChunk(
                        module=self.module,
                        field_name=field_name,
                        delta="",
                        content=value_str,
                        is_complete=True,
                    )
                )
                self.completed_fields.add(field_name)

        # Finalize tool calls
        native_tool_calls = self.adapter.finalize_tool_calls(self.tool_calls)

        # Get completion text
        completion_text = "".join(self.full_completion)

        return outputs, native_tool_calls, completion_text


class ChatAdapter:
    """Adapter for formatting signatures into OpenAI chat messages.

    This adapter converts Signature inputs into properly formatted
    chat messages and parses LLM responses back into structured outputs.

    The adapter handles both streaming and non-streaming responses:
    - For streaming: Call process_message() for each chunk, then finalize()
    - For non-streaming: Call process_message() once with the complete response
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._streaming_parser: StreamingParser | None = None

    def _get_or_create_parser(
        self,
        module: Any,
        signature: type[Signature],
    ) -> StreamingParser:
        """Get existing parser or create a new one.

        Args:
            module: Module instance
            signature: The signature defining expected outputs

        Returns:
            StreamingParser instance
        """
        if self._streaming_parser is None:
            self._streaming_parser = StreamingParser(self, module, signature)
        return self._streaming_parser

    def reset_parser(self) -> None:
        """Reset the streaming parser for a new request."""
        self._streaming_parser = None

    async def process_chunk(
        self,
        chunk: ChatCompletionChunk,
        module: Any,
        signature: type[Signature],
    ) -> Any:
        """Process an LLM streaming chunk.

        This method processes streaming chunks and yields StreamEvent objects.
        After processing all chunks, call finalize() to get validated outputs.

        Args:
            chunk: ChatCompletionChunk from streaming LLM
            module: Module instance
            signature: Signature defining expected outputs

        Yields:
            StreamEvent objects (ThoughtStreamChunk, OutputStreamChunk, etc.)
        """

        parser = self._get_or_create_parser(module, signature)
        async for event in parser.process_chunk(chunk):
            yield event

    async def finalize(
        self,
        signature: type[Signature],
    ) -> tuple[dict[str, Any], list[Any], str]:
        """Finalize streaming response and validate outputs.

        Must be called after processing all streaming chunks.

        Args:
            signature: Signature defining expected outputs

        Returns:
            Tuple of (outputs, native_tool_calls, completion_text)

        Raises:
            AdapterParseError: If outputs don't match signature or parsing fails
        """
        if self._streaming_parser is None:
            raise RuntimeError("No streaming parser available - call process_message first")

        (
            outputs,
            native_tool_calls,
            completion_text,
        ) = await self._streaming_parser.finalize()

        # Validate outputs match signature
        self.validate_outputs(signature, outputs, native_tool_calls, completion_text)

        # Reset for next request
        self.reset_parser()

        return outputs, native_tool_calls, completion_text

    def validate_outputs(
        self,
        signature: type[Signature],
        outputs: dict[str, Any],
        native_tool_calls: list[Any],
        completion_text: str,
    ) -> None:
        """Validate that outputs match the signature.

        Args:
            signature: Signature defining expected outputs
            outputs: Parsed output fields
            native_tool_calls: Tool calls from LLM
            completion_text: Raw completion text

        Raises:
            AdapterParseError: If outputs don't match signature
        """

        # If no tool calls, outputs must match signature exactly
        if not native_tool_calls and outputs.keys() != signature.get_output_fields().keys():
            raise AdapterParseError(
                adapter_name=self.__class__.__name__,
                signature=signature,
                lm_response=completion_text,
                parsed_result=outputs,
            )

    def split_reasoning_and_content_delta(
        self,
        response_or_chunk: ChatCompletion | ChatCompletionChunk,
    ) -> tuple[str, str]:
        """Split reasoning and content delta from a streaming chunk.

        This handles provider-specific reasoning formats:
        - OpenAI: choice.delta.reasoning (structured field)
        - AWS Bedrock: <reasoning>...</reasoning> tags in content
        - Other providers: may use different formats

        Args:
            response_or_chunk: ChatCompletion or ChatCompletionChunk from streaming LLM

        Returns:
            Tuple of (reasoning_delta, content_delta) where:
            - reasoning_delta: New reasoning content in this chunk
            - content_delta: Content excluding reasoning
        """
        if isinstance(response_or_chunk, ChatCompletion):
            message = response_or_chunk.choices[0].message
            message_content = message.content or ""
            reasoning_delta = getattr(message, "reasoning", None) or ""
        else:
            delta = response_or_chunk.choices[0].delta
            message_content = delta.content or ""
            reasoning_delta = getattr(delta, "reasoning", None) or ""

        # For some providers (like AWS Bedrock), reasoning is returned as
        # <reasoning> tags inside choice.delta.content instead of choice.delta.reasoning
        remaining_content = message_content
        if match := re.search(
            r"<reasoning>(.*?)</reasoning>(.*)",
            message_content,
            re.DOTALL | re.IGNORECASE,
        ):
            reasoning_delta = match.group(1).strip()
            remaining_content = match.group(2).strip()

        return reasoning_delta, remaining_content

    def process_tool_call_deltas(
        self,
        tool_calls_accumulator: dict[int, dict[str, Any]],
        delta_tool_calls: list[Any],
    ) -> None:
        """Process tool call deltas from streaming response.

        Accumulates tool call information across multiple chunks, handling
        incremental updates to tool call names and arguments.

        Args:
            tool_calls_accumulator: Dictionary mapping tool call index to accumulated data
            delta_tool_calls: List of tool call delta objects from the current chunk
        """
        for tc_delta in delta_tool_calls:
            idx = tc_delta.index
            if idx not in tool_calls_accumulator:
                tool_calls_accumulator[idx] = {
                    "id": tc_delta.id,
                    "type": tc_delta.type,
                    "function": {"name": "", "arguments": ""},
                }

            if tc_delta.function:
                if tc_delta.function.name:
                    tool_calls_accumulator[idx]["function"]["name"] += tc_delta.function.name
                if tc_delta.function.arguments:
                    tool_calls_accumulator[idx]["function"]["arguments"] += (
                        tc_delta.function.arguments
                    )

    def finalize_tool_calls(self, tool_calls_accumulator: dict[int, dict[str, Any]]) -> list[Any]:
        """Convert accumulated tool calls to ToolCall objects.

        Parses JSON arguments and creates properly formatted ToolCall instances.

        Args:
            tool_calls_accumulator: Dictionary of accumulated tool call data

        Returns:
            List of ToolCall objects

        Raises:
            AdapterParseError: If tool call arguments are not valid JSON
        """

        native_tool_calls = []
        for tc in tool_calls_accumulator.values():
            try:
                arguments = (
                    json.loads(tc["function"]["arguments"])
                    if isinstance(tc["function"]["arguments"], str)
                    else tc["function"]["arguments"]
                )
            except json.JSONDecodeError as exc:
                raise AdapterParseError(
                    adapter_name=self.__class__.__name__,
                    signature=None,  # Tool calls don't have a signature
                    lm_response=tc["function"]["arguments"],
                    message=f"Failed to parse tool call {tc['id']} arguments as JSON: {exc}",
                ) from exc

            native_tool_calls.append(
                ToolCall(
                    call_id=tc["id"],
                    name=tc["function"]["name"],
                    args=arguments,
                )
            )

        return native_tool_calls

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
        """Format signature instructions and field descriptions for system message.

        This now only includes the task description and input/output field descriptions,
        without the output formatting structure (which is moved to the user message).

        Args:
            signature: The signature to format

        Returns:
            Formatted instruction string for system message
        """
        parts = []

        instructions = signature.get_instructions()
        if instructions:
            parts.append(instructions)

        input_fields = [f"`{name}`" for name in signature.get_input_fields().keys()]
        output_fields = [f"`{name}`" for name in signature.get_output_fields().keys()]
        parts.append(
            f"Given the input fields: {', '.join(input_fields)}, produce the output fields: {', '.join(output_fields)}."
        )

        return "\n".join(parts).strip()

    def format_output_instructions(self, signature: type[Signature]) -> str:
        """Format instructions for how to structure output fields in JSON.

        This generates the part that tells the LLM how to respond with output fields
        as a JSON object.

        Args:
            signature: The signature defining expected outputs

        Returns:
            Formatted output instructions string
        """
        output_fields = signature.get_output_fields()
        if not output_fields:
            return ""

        parts = []
        parts.append("\n\nRespond with a JSON object containing the following fields:\n")

        # List all required fields
        for name, field_info in output_fields.items():
            type_hint = translate_field_type(name, field_info)
            # Extract constraint if exists
            constraint = ""
            if "# note:" in type_hint:
                constraint = " - " + type_hint.split("# note:", 1)[1].strip()

            type_name = (
                getattr(field_info.annotation, "__name__", "string")
                if field_info.annotation
                else "string"
            )
            parts.append(f"- `{name}`: {type_name}{constraint}\n")

        parts.append("\nReturn ONLY valid JSON with no additional text or markdown formatting.")

        return "".join(parts)

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

        for name, _ in input_fields.items():
            if name in inputs:
                value = inputs[name]
                formatted = format_value(value)
                parts.append(f"[[ ## {name} ## ]]\n{formatted}")

        return "\n\n".join(parts)

    def format_user_request(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> str:
        """Format complete user request with inputs and output instructions.

        This combines the input values with instructions on how to format outputs,
        creating a complete user message that tells the LLM what data it has and
        how to respond.

        Args:
            signature: The signature defining inputs and outputs
            inputs: Dictionary of input values

        Returns:
            Formatted user request string combining inputs + output instructions
        """
        try:
            formatted_inputs = self.format_inputs(signature, inputs)
        except Exception as e:
            raise ValueError(f"Failed to format inputs: {e}") from e
        output_instructions = self.format_output_instructions(signature)

        return formatted_inputs + output_instructions

    def parse_outputs(
        self,
        signature: type[Signature],
        completion: str,
    ) -> dict[str, Any]:
        """Parse LLM completion into structured outputs.

        Expects JSON format matching the signature's output fields.

        Args:
            signature: The signature defining expected outputs
            completion: Raw completion string from LLM (should be JSON)

        Returns:
            Dictionary of parsed output values

        Raises:
            AdapterParseError: If completion is not valid JSON
        """

        output_fields = signature.get_output_fields()

        # Handle empty completion (tool calls without content)
        if not completion or completion.strip() == "":
            return {}

        # Parse JSON completion
        try:
            outputs = json.loads(completion)
        except json.JSONDecodeError as e:
            raise AdapterParseError(
                adapter_name=self.__class__.__name__,
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse JSON output: {e}",
            ) from e

        if not isinstance(outputs, dict):
            raise AdapterParseError(
                adapter_name=self.__class__.__name__,
                signature=signature,
                lm_response=completion,
                message=f"Expected JSON object, got {type(outputs).__name__}",
            )

        parsed_outputs: dict[str, Any] = {}
        for field_name, field_info in output_fields.items():
            if field_name in outputs:
                value = outputs[field_name]
                field_type = field_info.annotation

                # Parse value according to field type
                try:
                    # Check if field type is a Pydantic model
                    if (
                        field_type
                        and isinstance(field_type, type)
                        and issubclass(field_type, BaseModel)
                    ):
                        # Convert dict to Pydantic model
                        if isinstance(value, dict):
                            parsed_outputs[field_name] = field_type.model_validate(value)
                        else:
                            # Try parsing as string
                            parsed_outputs[field_name] = parse_value(str(value), field_type)
                    elif isinstance(value, str):
                        # String value - parse according to type
                        parsed_outputs[field_name] = parse_value(value, field_type)  # type: ignore[arg-type]
                    else:
                        # Value is already correct type (int, float, list, dict, etc.)
                        parsed_outputs[field_name] = value
                except Exception:
                    # Fallback: keep original value
                    parsed_outputs[field_name] = value

        return parsed_outputs

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
