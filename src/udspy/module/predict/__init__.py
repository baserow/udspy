"""Predict module for LLM predictions based on signatures."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import regex as re
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from udspy.adapter import ChatAdapter
from udspy.callback import with_callbacks
from udspy.decorators import suspendable
from udspy.exceptions import AdapterParseError
from udspy.history import History
from udspy.module.base import Module
from udspy.module.predict.execution import execute_tool_calls
from udspy.module.predict.messages import (
    build_initial_messages,
    update_history_with_prediction,
)
from udspy.module.predict.stream_processing import (
    process_content_delta,
    process_reasoning_delta,
    process_tool_call_delta,
)
from udspy.module.predict.validators import check_valid_outputs_or_raise, validate_inputs
from udspy.settings import settings
from udspy.signature import Signature
from udspy.streaming import (
    OutputStreamChunk,
    Prediction,
    StreamEvent,
    ThoughtStreamChunk,
)
from udspy.tool import Tool, ToolCall

logger = logging.getLogger(__name__)


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
            if isinstance(event, OutputStreamChunk):
                print(event.delta, end="", flush=True)
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        *,
        tools: list[Tool] | None = None,
        max_turns: int = 10,
        adapter: ChatAdapter | None = None,
        model: str | None = None,
        **kwargs: Any,
    ):
        """Initialize a Predict module.

        Args:
            signature: Signature defining inputs and outputs, or a string in
                      format "inputs -> outputs" (e.g., "question -> answer")
            model: Model name (overrides global default)
            tools: List of tool functions (decorated with @tool) or Pydantic models
            max_turns: Maximum number of LLM calls for tool execution loop (default: 10)
            adapter: Custom adapter (defaults to ChatAdapter)
            **kwargs: Additional arguments for chat completion (temperature, callbacks, etc.)
        """
        if isinstance(signature, str):
            signature = Signature.from_string(signature)

        self.signature = signature
        self._model = model
        self._kwargs = kwargs
        self.max_turns = max_turns
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.adapter = adapter or ChatAdapter()

        self.init_module(tools=tools)

    @property
    def model(self) -> str | None:
        """Get the model name override, or None to use LM's default."""
        return self._model

    @property
    def kwargs(self) -> dict[str, Any]:
        return {**settings.default_kwargs, **self._kwargs}

    def init_module(self, tools: list[Any] | None = None) -> None:
        """Initialize or reinitialize Predict with new tools.

        This method rebuilds the tools dictionary and regenerates tool schemas.
        It's designed to be called from module callbacks to dynamically modify
        available tools during execution.

        Args:
            tools: New tools to initialize with. Can be:
                - Functions decorated with @tool
                - Tool instances
                - None to clear all tools

        Example:
            ```python
            from udspy import module_callback

            @module_callback
            def add_specialized_tools(context):
                # Get current tools
                current_tools = list(context.module.tools.values())

                # Add new tools
                new_tools = [weather_tool, calendar_tool]

                # Reinitialize with all tools
                context.module.init_module(tools=current_tools + new_tools)

                return "Added weather and calendar tools"
            ```
        """
        self._init_tools(tools or [])

    def _init_tools(self, tools: list[Any]) -> None:
        """Initialize tools dictionary with provided tools.

        Args:
            tools: List of tools (functions or Tool instances)
        """
        tool_list = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        self.tools = {tool.name: tool for tool in tool_list if tool.name}
        self._build_tool_schemas()

    def _build_tool_schemas(self) -> None:
        """Build OpenAI tool schemas from current tools."""
        self.tool_schemas = [self.adapter.format_tool_schema(tool) for tool in self.tools.values()]

    @suspendable
    @with_callbacks
    async def aexecute(
        self,
        *,
        stream: bool = False,
        auto_execute_tools: bool = True,
        history: History | None = None,
        **inputs: Any,
    ) -> Prediction:
        """Core execution method - handles both streaming and non-streaming.

        This is the single implementation point for LLM interaction. It always
        returns a Prediction, and emits events to the queue if one is active.

        Args:
            stream: If True, request streaming from OpenAI. If False, use regular API.
            auto_execute_tools: If True, automatically execute tools and continue.
                If False, return Prediction with tool_calls for manual handling.
            history: Optional History object for multi-turn conversations.
            **inputs: Input values matching the signature's input fields

        Returns:
            Final Prediction object (after all tool executions if auto_execute_tools=True)
        """
        from udspy.streaming import _stream_queue

        if history is None:
            history = History()

        queue = _stream_queue.get()
        should_emit = queue is not None

        self._validate_inputs(inputs)
        self._build_initial_messages(inputs, history)

        final_prediction = await self._aexecute(
            stream=stream,
            should_emit=should_emit,
            auto_execute_tools=auto_execute_tools,
            history=history,
        )

        if should_emit and queue:
            await queue.put(final_prediction)

        return final_prediction

    async def astream(
        self,
        *,
        resume_state: Any = None,
        auto_execute_tools: bool = True,
        history: History | None = None,
        **inputs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming method with optional automatic tool execution.

        Sets up streaming queue and yields events. Automatically handles multi-turn
        conversation when tools are present.

        Supports resuming from a ConfirmationRequired exception by providing
        resume_state. This enables streaming with confirmation handling.

        Args:
            resume_state: Optional ResumeState containing exception and user response.
            auto_execute_tools: If True, automatically execute tools and continue.
                If False, return Prediction with tool_calls for manual handling.
            history: Optional History object for multi-turn conversations.
            **inputs: Input values matching the signature's input fields

        Yields:
            StreamEvent objects (OutputStreamChunk, Prediction, custom events)
        """
        async for event in super().astream(
            resume_state=resume_state,
            auto_execute_tools=auto_execute_tools,
            history=history,
            **inputs,
        ):
            yield event

    def _validate_inputs(self, inputs: dict[str, Any]) -> None:
        """Validate that all required inputs are provided."""
        validate_inputs(self.signature, inputs)

    def _build_initial_messages(self, inputs: dict[str, Any], history: History) -> None:
        """Build initial messages from inputs and optional history.

        Args:
            inputs: Input values from user
            history: History object with existing conversation
        """
        build_initial_messages(self.adapter, self.signature, inputs, history)

    async def _aexecute(
        self,
        stream: bool,
        should_emit: bool,
        auto_execute_tools: bool,
        history: History,
    ) -> Prediction:
        """Execute multi-turn conversation with optional automatic tool execution.

        This is the core execution loop that handles both streaming and non-streaming.
        It always returns a final Prediction, and emits events if should_emit is True.

        Args:
            stream: If True, request streaming from OpenAI
            should_emit: If True, emit events to active queue
            auto_execute_tools: If True, automatically execute tools. If False,
                return after first tool call.
            history: Optional History object to update with conversation

        Returns:
            Final Prediction object
        """
        prediction: Prediction | None = None

        for turn in range(self.max_turns):
            prediction = await self._aexecute_one_turn(
                history.messages, turn, stream=stream, should_emit=should_emit
            )

            if not auto_execute_tools or not self.tools or not prediction.native_tool_calls:
                break

            await self._aexecute_tool_calls(prediction.native_tool_calls, history)
        else:
            if prediction is not None and not prediction.is_final():
                raise RuntimeError(f"Max turns ({self.max_turns}) reached without final answer")

        if prediction is None:
            raise RuntimeError("No prediction generated")

        self._update_history_with_prediction(history, prediction)
        return prediction

    async def _aexecute_one_turn(
        self, messages: list[dict[str, Any]], turn: int, stream: bool, should_emit: bool
    ) -> Prediction:
        """Execute one LLM turn (streaming or non-streaming).

        Args:
            messages: Conversation messages
            turn: Current turn number (0-indexed)
            stream: If True, request streaming from OpenAI
            should_emit: If True, emit events to active queue

        Returns:
            Prediction object for this turn
        """
        completion_kwargs: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
            "tools": self.tool_schemas,
            **self.kwargs,
        }

        # Only pass model if explicitly set (otherwise LM uses its default)
        if self.model is not None:
            completion_kwargs["model"] = self.model

        func = self._astream if stream else self._aforward
        return await func(completion_kwargs, should_emit)

    async def _aexecute_tool_calls(
        self,
        native_tool_calls: list[ToolCall],
        history: History,
    ) -> None:
        """Execute tool calls and add results to messages.

        Args:
            tool_calls: List of tool calls to execute
            history: History object to update
        """
        await execute_tool_calls(self, self.tools, native_tool_calls, history)

    def _update_history_with_prediction(self, history: History, prediction: Prediction) -> None:
        """Update history with assistant's prediction.

        Args:
            history: History object to update
            prediction: Prediction from assistant
        """
        update_history_with_prediction(self.signature, history, prediction)

    async def aforward(
        self,
        *,
        resume_state: Any = None,
        auto_execute_tools: bool = True,
        history: History | None = None,
        **inputs: Any,
    ) -> Prediction:
        """Async non-streaming method. Returns the final Prediction.

        Calls aexecute() with streaming disabled. If called from within a
        streaming context (i.e., another module is streaming), events will
        still be emitted to the active queue.

        Supports resuming from a ConfirmationRequired exception by providing
        resume_state. This enables loop-based confirmation handling.

        When tools are used with auto_execute_tools=True (default), this returns
        the LAST prediction (after tool execution), not the first one (which might
        only contain tool_calls). When auto_execute_tools=False, returns the first
        Prediction with tool_calls for manual handling.

        Args:
            resume_state: Optional ResumeState containing exception and user response.
            auto_execute_tools: If True, automatically execute tools and return
                final answer. If False, return Prediction with tool_calls for
                manual execution. Default: True.
            history: Optional History object for multi-turn conversations.
            **inputs: Input values for the module

        Returns:
            Final Prediction object (after all tool executions if auto_execute_tools=True)
        """
        return await self.aexecute(
            stream=False,
            resume_state=resume_state,
            auto_execute_tools=auto_execute_tools,
            history=history,
            **inputs,
        )

    def _process_tool_call_delta(
        self, tool_calls: dict[int, dict[str, Any]], delta_tool_calls: list[Any]
    ) -> None:
        """Process tool call deltas and accumulate them.

        Args:
            tool_calls: Dictionary to accumulate tool calls in
            delta_tool_calls: List of tool call deltas from the chunk
        """
        process_tool_call_delta(tool_calls, delta_tool_calls)

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
        return await process_content_delta(
            self,
            delta,
            acc_delta,
            current_field,
            accumulated_content,
            output_fields,
            field_pattern,
            queue,
        )

    def _execute_lm_callbacks(
        self,
        stage: str,
        call_id: str,
        inputs: dict | None = None,
        outputs: dict | None = None,
        exception: Exception | None = None,
    ) -> None:
        """Execute LM callbacks for start/end events.

        Args:
            stage: "start" or "end"
            call_id: Unique call identifier
            inputs: Input parameters for LM call (for start)
            outputs: Output from LM call (for end)
            exception: Exception if LM call failed (for end)
        """
        from udspy.callback import BaseCallback

        global_callbacks = settings.get("callbacks", [])
        instance_callbacks = getattr(self, "callbacks", [])
        callbacks = global_callbacks + instance_callbacks

        for callback in callbacks:
            if not isinstance(callback, BaseCallback):
                continue

            try:
                if stage == "start" and inputs is not None:
                    callback.on_lm_start(call_id=call_id, instance=self, inputs=inputs)
                elif stage == "end":
                    callback.on_lm_end(call_id=call_id, outputs=outputs, exception=exception)
            except Exception as e:
                logger.warning(
                    f"Error in callback {callback.__class__.__name__}.on_lm_{stage}: {e}"
                )

    def _check_valid_outputs_or_raise(
        self,
        native_tool_calls: list[ToolCall],
        outputs: dict[str, Any],
        completion_text: str,
    ) -> None:
        """
        Check if the tool calls and outputs are valid; raise AdapterParseError if not.
        """
        check_valid_outputs_or_raise(
            self.adapter.__class__.__name__,
            self.signature,
            self.tools,
            native_tool_calls,
            outputs,
            completion_text,
        )

    @retry(
        retry=retry_if_exception_type(AdapterParseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=3),
    )
    async def _aforward(self, completion_kwargs: dict[str, Any], should_emit: bool) -> Prediction:
        """Process non-streaming LLM call with automatic retry on parse errors.

        Retries up to 2 times (3 total attempts) with exponential backoff (0.1-3s)
        when AdapterParseError occurs, giving the LLM multiple chances to format
        the response correctly.

        Args:
            completion_kwargs: Arguments for the completion API call
            should_emit: If True, emit events to active queue

        Returns:
            Prediction object
        """
        import uuid

        from udspy.streaming import _stream_queue

        call_id = uuid.uuid4().hex
        self._execute_lm_callbacks("start", call_id, inputs=completion_kwargs)

        outputs_dict = None
        exception = None

        try:
            response = await settings.lm.acomplete(**completion_kwargs)
            outputs_dict = {
                "response": (
                    response.model_dump() if hasattr(response, "model_dump") else str(response)
                )
            }

            message = response.choices[0].message  # type: ignore[union-attr]
            completion_text = message.content or ""
            native_tool_calls: list[ToolCall] = []
            for tc in message.tool_calls or []:
                try:
                    arguments = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    )
                except json.JSONDecodeError as exc:
                    raise AdapterParseError(
                        adapter_name=self.adapter.__class__.__name__,
                        signature=self.signature,
                        lm_response=tc.function.arguments,
                        parsed_result={
                            "error": f"Failed to parse tool call {tc.id} arguments as JSON."
                        },
                    ) from exc

                else:
                    native_tool_calls.append(
                        ToolCall(call_id=tc.id, name=tc.function.name, args=arguments)
                    )

            outputs = self.adapter.parse_outputs(self.signature, completion_text)
            self._check_valid_outputs_or_raise(native_tool_calls, outputs, completion_text)

        except Exception as e:
            exception = e
            raise
        finally:
            self._execute_lm_callbacks("end", call_id, outputs=outputs_dict, exception=exception)

        prediction = Prediction(native_tool_calls=native_tool_calls, **outputs)

        if should_emit and (queue := _stream_queue.get()) is not None:
            await queue.put(prediction)

        return prediction

    @retry(
        retry=retry_if_exception_type(AdapterParseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=3),
    )
    async def _astream(self, completion_kwargs: dict[str, Any], should_emit: bool) -> Prediction:
        """Process streaming LLM call with automatic retry on parse errors.

        Retries up to 2 times (3 total attempts) with exponential backoff (0.1-3s)
        when AdapterParseError occurs, giving the LLM multiple chances to format
        the response correctly.

        Args:
            completion_kwargs: Arguments for the completion API call
            should_emit: If True, emit events to active queue from context

        Returns:
            Prediction object
        """
        from udspy.streaming import _stream_queue

        active_queue = _stream_queue.get()

        if should_emit and active_queue is not None:
            return await self._process_llm_stream(
                active_queue,
                completion_kwargs,
                emit_sentinel=False,
                emit_prediction=False,
            )
        else:
            queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
            llm_task = asyncio.create_task(
                self._process_llm_stream(queue, completion_kwargs, emit_sentinel=True)
            )

            prediction: Prediction | None = None
            while True:
                event = await queue.get()
                if event is None:
                    break

                if isinstance(event, Prediction):
                    prediction = event

            await llm_task

            if prediction is None:
                raise RuntimeError("No prediction generated from stream")

            return prediction

    async def _process_llm_stream(
        self,
        queue: asyncio.Queue[StreamEvent | None],
        completion_kwargs: dict[str, Any],
        emit_sentinel: bool = True,
        emit_prediction: bool = True,
    ) -> Prediction:
        """Background task to process LLM stream and put events in queue.

        Args:
            queue: Event queue to put events in
            completion_kwargs: Arguments for the completion API call
            emit_sentinel: If True, emit None sentinel at the end
            emit_prediction: If True, emit final Prediction to queue

        Returns:
            Final Prediction object
        """
        import uuid

        call_id = uuid.uuid4().hex
        self._execute_lm_callbacks("start", call_id, inputs=completion_kwargs)

        outputs_dict = None
        exception = None

        try:
            stream: AsyncStream[ChatCompletionChunk] = await settings.lm.acomplete(  # type: ignore[assignment]
                **completion_kwargs
            )

            output_fields = self.signature.get_output_fields()
            field_pattern = re.compile(
                r"\[\[ ## (?P<field>\w+) ## \]\]\s*(?P<content>.*)", re.DOTALL
            )
            current_field: str | None = None
            accumulated_content: dict[str, list[str]] = {name: [] for name in output_fields}
            full_completion: list[str] = []
            acc_delta: str = ""
            tool_calls: dict[int, dict[str, Any]] = {}

            reasoning = ThoughtStreamChunk(self, "thought", "", "", is_complete=False)

            async for chunk in stream:
                choice = chunk.choices[0]

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

                if current_field and current_field in output_fields:
                    field_content = "".join(accumulated_content[current_field])
                    await queue.put(
                        OutputStreamChunk(self, current_field, "", field_content, is_complete=True)
                    )
                elif choice.delta.tool_calls:
                    self._process_tool_call_delta(tool_calls, choice.delta.tool_calls)
                elif not reasoning.is_complete:
                    reasoning = await self._process_reasoning_delta(reasoning, choice, queue)

            completion_text = "".join(full_completion)
            outputs = self.adapter.parse_outputs(self.signature, completion_text)
            native_tool_calls = []
            for tc in tool_calls.values():
                try:
                    arguments = (
                        json.loads(tc["function"]["arguments"])
                        if isinstance(tc["function"]["arguments"], str)
                        else tc["function"]["arguments"]
                    )

                except json.JSONDecodeError as exc:
                    raise AdapterParseError(
                        adapter_name=self.adapter.__class__.__name__,
                        signature=self.signature,
                        lm_response=arguments,
                        parsed_result={
                            "error": f"Failed to parse tool call {tc['id']} arguments as JSON."
                        },
                    ) from exc
                else:
                    native_tool_calls.append(
                        ToolCall(
                            call_id=tc["id"],
                            name=tc["function"]["name"],
                            args=arguments,
                        )
                    )

            self._check_valid_outputs_or_raise(native_tool_calls, outputs, completion_text)

            prediction = Prediction(native_tool_calls=native_tool_calls, **outputs)
            if emit_prediction:
                await queue.put(prediction)

            outputs_dict = {
                "prediction": (
                    prediction.model_dump()
                    if hasattr(prediction, "model_dump")
                    else str(prediction)
                )
            }
            return prediction

        except Exception as e:
            import traceback

            exception = e
            error_event = type(
                "StreamError",
                (StreamEvent,),
                {"error": str(e), "traceback": traceback.format_exc()},
            )()
            await queue.put(error_event)
            raise
        finally:
            self._execute_lm_callbacks("end", call_id, outputs=outputs_dict, exception=exception)
            if emit_sentinel:
                await queue.put(None)

    async def _process_reasoning_delta(
        self,
        reasoning: ThoughtStreamChunk,
        choice: Any,
        queue: asyncio.Queue[StreamEvent | None],
    ) -> ThoughtStreamChunk:
        """Process reasoning delta from choice.

        Args:
            reasoning: Current reasoning chunk state
            choice: Choice object from streaming chunk
            queue: Event queue to emit chunks to

        Returns:
            Updated ThoughtStreamChunk
        """
        return await process_reasoning_delta(self, reasoning, choice, queue)

    async def asuspend(self, exception: Any) -> Any:
        """Suspend execution and save state for Predict module.

        Saves the current conversation state (messages, turn number, tool state)
        needed to resume execution after user confirmation.

        Args:
            exception: The ConfirmationRequired exception that was raised

        Returns:
            Saved state dict containing messages, turn, and tool info
        """
        # The exception itself contains the context we need
        # No additional state needed for Predict - the decorator handles it
        return exception

    async def aresume(self, user_response: str, saved_state: Any) -> Prediction:  # noqa: ARG002
        """Resume execution after user confirmation.

        Args:
            user_response: The user's response to the confirmation request
            saved_state: State saved by asuspend() (the ConfirmationRequired exception)

        Returns:
            Final Prediction after resuming execution

        Raises:
            NotImplementedError: Predict doesn't currently support resumption.
                This will be implemented when tool confirmation is added.
        """
        # TODO: Implement proper resumption for Predict
        # This requires saving/restoring the conversation state,
        # current turn, and tool execution context
        raise NotImplementedError(
            "Predict module doesn't yet support suspend/resume. "
            "Use ReAct module for confirmation support."
        )
