"""Predict module for LLM predictions based on signatures."""

import asyncio
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Optional

import regex as re
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from udspy.adapter import ChatAdapter
from udspy.module.base import Module
from udspy.settings import settings
from udspy.signature import Signature
from udspy.streaming import Prediction, StreamChunk, StreamEvent, _stream_queue

if TYPE_CHECKING:
    from udspy.history import History


class Predict(Module):
    """Module for making LLM predictions based on a signature.

    This is an async-first module. The core method is `astream()` which yields
    StreamEvent objects. Use `aforward()` for async non-streaming, or `forward()`
    for sync usage.

    Example:
        ```python
        from udspy import Predict, Signature, InputField, OutputField

        class QA(Signature):
            '''Answer questions.'''
            question: str = InputField()
            answer: str = OutputField()

        predictor = Predict(QA)

        # Sync usage
        result = predictor(question="What is 2+2?")
        print(result.answer)

        # Async non-streaming
        result = await predictor.aforward(question="What is 2+2?")

        # Async streaming
        async for event in predictor.astream(question="What is 2+2?"):
            if isinstance(event, StreamChunk):
                print(event.delta, end="", flush=True)
        ```
    """

    def __init__(
        self,
        signature: type[Signature],
        *,
        model: str | None = None,
        tools: list[Any] | None = None,
        max_turns: int = 5,
        adapter: ChatAdapter | None = None,
        **kwargs: Any,
    ):
        """Initialize a Predict module.

        Args:
            signature: Signature defining inputs and outputs
            model: Model name (overrides global default)
            tools: List of tool functions (decorated with @tool) or Pydantic models
            max_turns: Maximum number of LLM calls for tool execution loop (default: 5)
            adapter: Custom adapter (defaults to ChatAdapter)
            **kwargs: Additional arguments for chat completion (temperature, etc.)
        """
        from udspy.tool import Tool

        self.signature = signature
        self.model = model or settings.default_model
        self.max_turns = max_turns
        self.adapter = adapter or ChatAdapter()
        self.kwargs = {**settings.default_kwargs, **kwargs}

        # Process tools - separate Tool objects from Pydantic models
        self.tool_callables: dict[str, Tool] = {}
        self.tool_schemas: list[Any] = []

        for tool in tools or []:
            if isinstance(tool, Tool):
                # Tool decorator - store both callable and schema
                self.tool_callables[tool.name] = tool
                self.tool_schemas.append(tool)
            else:
                # Pydantic model - just schema (no automatic execution)
                self.tool_schemas.append(tool)

    async def astream(
        self, *, auto_execute_tools: bool = True, history: Optional["History"] = None, **inputs: Any
    ) -> AsyncGenerator[StreamEvent, None]:
        """Core async streaming method with optional automatic tool execution.

        Yields StreamEvent objects. When auto_execute_tools=True (default),
        automatically handles multi-turn conversation when tools are present.
        When False, returns Prediction with tool_calls for manual handling.

        Args:
            auto_execute_tools: If True, automatically execute tools and continue
                conversation. If False, return Prediction with tool_calls for
                manual execution. Default: True.
            history: Optional History object for multi-turn conversations. When
                provided, conversation history is automatically managed.
            **inputs: Input values matching the signature's input fields

        Yields:
            StreamEvent objects (StreamChunk, Prediction, custom events)

        Raises:
            ValueError: If required inputs are missing
        """
        # Validate and build initial messages
        self._validate_inputs(inputs)
        messages = self._build_initial_messages(inputs, history)

        # Multi-turn loop for tool execution
        async for event in self._execute_with_tools(messages, auto_execute_tools, history):
            yield event

    def _validate_inputs(self, inputs: dict[str, Any]) -> None:
        """Validate that all required inputs are provided."""
        input_fields = self.signature.get_input_fields()
        for field_name in input_fields:
            if field_name not in inputs:
                raise ValueError(f"Missing required input field: {field_name}")

    def _build_initial_messages(
        self, inputs: dict[str, Any], history: Any = None
    ) -> list[dict[str, Any]]:
        """Build initial messages from inputs and optional history.

        Args:
            inputs: Input values from user
            history: Optional History object with existing conversation

        Returns:
            List of messages including history (if provided) and new user input
        """
        from udspy.history import History

        messages: list[dict[str, Any]] = []

        # Start with history if provided
        if history is not None:
            if isinstance(history, History):
                # Copy existing messages from history
                messages.extend(history.messages)
            else:
                raise TypeError(f"history must be a History object, got {type(history)}")

        # Add system message if not in history
        if not messages or messages[0]["role"] != "system":
            messages.insert(
                0, {"role": "system", "content": self.adapter.format_instructions(self.signature)}
            )

        # Add new user input
        user_content = self.adapter.format_inputs(self.signature, inputs)
        user_msg = {"role": "user", "content": user_content}
        messages.append(user_msg)

        # Also add to history if provided
        if history is not None and isinstance(history, History):
            history.messages.append(user_msg)

        return messages

    async def _execute_with_tools(
        self, messages: list[dict[str, Any]], auto_execute_tools: bool, history: Any = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute multi-turn conversation with optional automatic tool execution.

        Args:
            messages: Conversation messages
            auto_execute_tools: If True, automatically execute tools. If False,
                return after first tool call.
            history: Optional History object to update with conversation
        """
        for turn in range(self.max_turns):
            # Stream one LLM turn
            final_prediction = None
            async for event in self._stream_one_turn(messages, turn):
                if isinstance(event, Prediction):
                    final_prediction = event
                yield event

            # Update history with prediction if provided
            if history is not None and final_prediction:
                self._update_history_with_prediction(history, final_prediction)

            # Check if we have tool calls
            if not (final_prediction and "tool_calls" in final_prediction):
                break  # No tools requested, we're done

            # If not auto-executing, stop here and return the tool_calls
            if not auto_execute_tools:
                break  # User will handle tool calls manually

            # Check if we can execute (need tool_callables)
            if not self.tool_callables:
                break  # No executables, stop here

            # Execute tools and add results to messages
            self._execute_tool_calls(messages, final_prediction.tool_calls, history)

        # Check if we exceeded max turns
        if turn >= self.max_turns - 1 and final_prediction and "tool_calls" in final_prediction:
            raise RuntimeError(f"Max turns ({self.max_turns}) reached without final answer")

    async def _stream_one_turn(
        self, messages: list[dict[str, Any]], turn: int
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream one LLM turn."""
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.kwargs,
        }

        # Add tools on first turn
        if turn == 0 and self.tool_schemas:
            tool_schemas = self.adapter.format_tool_schemas(self.tool_schemas)
            completion_kwargs["tools"] = tool_schemas

        async for event in self._stream_with_queue(completion_kwargs):
            yield event

    def _execute_tool_calls(
        self, messages: list[dict[str, Any]], tool_calls: list[dict[str, Any]], history: Any = None
    ) -> None:
        """Execute tool calls and add results to messages.

        Args:
            messages: Conversation messages
            tool_calls: List of tool calls to execute
            history: Optional History object to update
        """
        import json

        # Add assistant message with tool calls
        assistant_msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)

        # Update history if provided
        if history is not None:
            from udspy.history import History

            if isinstance(history, History):
                history.messages.append(assistant_msg)

        # Execute each tool
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            arguments = json.loads(tool_call["arguments"])

            if tool_name in self.tool_callables:
                try:
                    result = self.tool_callables[tool_name](**arguments)
                    content = str(result)
                except Exception as e:
                    content = f"Error executing tool: {e}"
            else:
                content = f"Error: Tool {tool_name} not found"

            tool_msg = {"role": "tool", "tool_call_id": tool_call["id"], "content": content}
            messages.append(tool_msg)

            # Update history if provided
            if history is not None:
                from udspy.history import History

                if isinstance(history, History):
                    history.messages.append(tool_msg)

    def _update_history_with_prediction(self, history: Any, prediction: Prediction) -> None:
        """Update history with assistant's prediction.

        Args:
            history: History object to update
            prediction: Prediction from assistant
        """
        from udspy.history import History

        if not isinstance(history, History):
            return

        # Build content from prediction output fields
        output_fields = self.signature.get_output_fields()
        content_parts = []

        for field_name in output_fields:
            if hasattr(prediction, field_name):
                value = getattr(prediction, field_name)
                if value:
                    content_parts.append(f"[[ ## {field_name} ## ]]\n{value}")

        content = "\n".join(content_parts) if content_parts else ""

        # Check if this prediction has tool_calls
        if hasattr(prediction, "tool_calls") and prediction.tool_calls:
            # Don't add to history yet - will be added in _execute_tool_calls
            pass
        else:
            # Regular assistant message
            history.add_assistant_message(content)

    async def aforward(
        self, *, auto_execute_tools: bool = True, history: Any = None, **inputs: Any
    ) -> Prediction:
        """Async non-streaming method. Returns the final Prediction.

        When tools are used with auto_execute_tools=True (default), this returns
        the LAST prediction (after tool execution), not the first one (which might
        only contain tool_calls). When auto_execute_tools=False, returns the first
        Prediction with tool_calls for manual handling.

        Args:
            auto_execute_tools: If True, automatically execute tools and return
                final answer. If False, return Prediction with tool_calls for
                manual execution. Default: True.
            history: Optional History object for multi-turn conversations.
            **inputs: Input values for the module

        Returns:
            Final Prediction object (after all tool executions if auto_execute_tools=True)
        """
        final_prediction: Prediction | None = None
        async for event in self.astream(
            auto_execute_tools=auto_execute_tools, history=history, **inputs
        ):
            if isinstance(event, Prediction):
                final_prediction = event  # Keep updating to get the last one

        if final_prediction is None:
            raise RuntimeError(f"{self.__class__.__name__}.astream() did not yield a Prediction")

        return final_prediction

    def _process_tool_call_delta(
        self, tool_calls: dict[int, dict[str, Any]], delta_tool_calls: list[Any]
    ) -> None:
        """Process tool call deltas and accumulate them.

        Args:
            tool_calls: Dictionary to accumulate tool calls in
            delta_tool_calls: List of tool call deltas from the chunk
        """
        for tool_call in delta_tool_calls:
            idx = tool_call.index
            if idx not in tool_calls:
                tool_calls[idx] = {
                    "id": tool_call.id or "",
                    "type": tool_call.type or "function",
                    "function": {
                        "name": tool_call.function.name if tool_call.function else "",
                        "arguments": "",
                    },
                }

            # Accumulate function arguments
            if tool_call.function and tool_call.function.arguments:
                tool_calls[idx]["function"]["arguments"] += tool_call.function.arguments

    async def _process_content_delta(
        self,
        delta: str,
        acc_delta: str,
        current_field: str | None,
        accumulated_content: dict[str, list[str]],
        output_fields: dict[str, Any],
        field_pattern: re.Pattern[str],
        queue: asyncio.Queue[StreamEvent | None],
    ) -> tuple[str, str | None]:
        """Process content delta and stream field chunks.

        Args:
            delta: New content delta
            acc_delta: Accumulated delta so far
            current_field: Current field being processed
            accumulated_content: Dictionary of accumulated content per field
            output_fields: Output fields from signature
            field_pattern: Regex pattern for field markers
            queue: Event queue to put chunks in

        Returns:
            Tuple of (updated acc_delta, updated current_field)
        """
        acc_delta += delta

        if not acc_delta:
            return acc_delta, current_field

        # Check for field markers
        match = field_pattern.search(acc_delta)
        if match:
            # Entering a new field
            if current_field:
                # Mark previous field as complete
                field_content = "".join(accumulated_content[current_field])
                await queue.put(
                    StreamChunk(self, current_field, "", field_content, is_complete=True)
                )

            current_field = match.group(1)
            # Remove the marker from content
            acc_delta = field_pattern.sub("", acc_delta)
            if acc_delta.startswith("\n"):
                acc_delta = acc_delta[1:]

        # Stream content for current field
        if (
            current_field
            and current_field in output_fields
            and not field_pattern.match(acc_delta, partial=True)
        ):
            accumulated_content[current_field].append(acc_delta)
            field_content = "".join(accumulated_content[current_field])
            await queue.put(
                StreamChunk(self, current_field, acc_delta, field_content, is_complete=False)
            )
            acc_delta = ""

        return acc_delta, current_field

    async def _process_llm_stream(
        self, queue: asyncio.Queue[StreamEvent | None], completion_kwargs: dict[str, Any]
    ) -> None:
        """Background task to process LLM stream and put events in queue.

        Args:
            queue: Event queue to put events in
            completion_kwargs: Arguments for the completion API call
        """
        try:
            # Make streaming API call
            client = settings.aclient
            stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
                **completion_kwargs
            )

            # Initialize streaming state
            output_fields = self.signature.get_output_fields()
            field_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")
            current_field: str | None = None
            accumulated_content: dict[str, list[str]] = {name: [] for name in output_fields}
            full_completion: list[str] = []
            acc_delta: str = ""
            tool_calls: dict[int, dict[str, Any]] = {}

            # Process stream
            async for chunk in stream:
                choice = chunk.choices[0]

                # Handle tool calls
                if choice.delta.tool_calls:
                    self._process_tool_call_delta(tool_calls, choice.delta.tool_calls)

                # Handle content
                delta = choice.delta.content or ""
                if delta:
                    full_completion.append(delta)
                    acc_delta, current_field = await self._process_content_delta(
                        delta,
                        acc_delta,
                        current_field,
                        accumulated_content,
                        output_fields,
                        field_pattern,
                        queue,
                    )

            # Mark last field as complete
            if current_field:
                field_content = "".join(accumulated_content[current_field])
                await queue.put(
                    StreamChunk(self, current_field, "", field_content, is_complete=True)
                )

            # Parse final outputs
            completion_text = "".join(full_completion)
            outputs = self.adapter.parse_outputs(self.signature, completion_text)

            # Add tool calls if present
            if tool_calls:
                outputs["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    }
                    for tc in tool_calls.values()
                ]

            # Put final prediction
            await queue.put(Prediction(**outputs))

        except Exception as e:
            # On error, put exception in queue
            import traceback

            error_event = type(
                "StreamError",
                (StreamEvent,),
                {"error": str(e), "traceback": traceback.format_exc()},
            )()
            await queue.put(error_event)
        finally:
            # Signal completion with sentinel
            await queue.put(None)

    async def _stream_with_queue(
        self, completion_kwargs: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Internal method to handle streaming with event queue.

        This sets up the event queue context and processes both LLM stream
        and tool events concurrently.
        """
        # Create event queue for this stream
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
        token = _stream_queue.set(queue)

        try:
            # Start background LLM processing
            llm_task = asyncio.create_task(self._process_llm_stream(queue, completion_kwargs))

            # Yield events from queue until sentinel
            while True:
                event = await queue.get()
                if event is None:  # Sentinel - stream complete
                    break
                yield event

            # Wait for background task to complete
            await llm_task

        finally:
            # Clean up context
            try:
                _stream_queue.reset(token)
            except (ValueError, LookupError):
                # Context was already cleaned up or created in different context
                # This can happen when the generator is closed due to an exception
                pass
