"""Predict module for LLM predictions based on signatures."""

import json
import logging
from typing import Any

import regex as re
from openai import AsyncStream, BaseModel
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
from udspy.module.callbacks import PredictContext, is_module_callback
from udspy.settings import settings
from udspy.signature import Signature
from udspy.streaming import (
    Prediction,
    StreamEvent,
    ThoughtStreamChunk,
    emit_event,
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
        from udspy.streaming import OutputStreamChunk
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
        if history is None:
            history = History()

        self._validate_inputs(inputs)
        self._build_initial_messages(inputs, history)

        return await self._aexecute(
            stream=stream,
            auto_execute_tools=auto_execute_tools,
            history=history,
        )

    def _validate_inputs(self, inputs: dict[str, Any]) -> None:
        """Validate that all required inputs are provided."""
        input_fields = self.signature.get_input_fields()
        for field_name in input_fields:
            if field_name not in inputs:
                raise ValueError(f"Missing required input field: {field_name}")

    def _build_initial_messages(self, inputs: dict[str, Any], history: History) -> None:
        """Build initial messages from inputs and optional history.

        Args:
            inputs: Input values from user
            history: History object with existing conversation
        """
        history.set_system_message(self.adapter.format_instructions(self.signature))
        history.add_user_message(self.adapter.format_user_request(self.signature, inputs))

    async def _aexecute(
        self,
        stream: bool,
        auto_execute_tools: bool,
        history: History,
    ) -> Prediction:
        """Execute multi-turn conversation with optional automatic tool execution.

        This is the core execution loop that handles both streaming and non-streaming.

        Args:
            stream: If True, request streaming from OpenAI
            auto_execute_tools: If True, automatically execute tools. If False,
                return after first tool call.
            history: Optional History object to update with conversation

        Returns:
            Final Prediction object
        """
        prediction: Prediction | None = None

        for turn in range(self.max_turns):
            prediction = await self._aexecute_one_turn(history.messages, turn, stream=stream)

            if not auto_execute_tools or not prediction.native_tool_calls:
                break

            await self._aexecute_tool_calls(prediction.native_tool_calls, history)
        else:
            if prediction is not None and not prediction.is_final:
                raise RuntimeError(f"Max turns ({self.max_turns}) reached without final answer")

        if prediction is None:
            raise RuntimeError("No prediction generated")

        self._update_history_with_prediction(history, prediction)
        return prediction

    async def _aexecute_one_turn(
        self, messages: list[dict[str, Any]], turn: int, stream: bool
    ) -> Prediction:
        """Execute one LLM turn (streaming or non-streaming).

        Args:
            messages: Conversation messages
            turn: Current turn number (0-indexed)
            stream: If True, request streaming from OpenAI

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
        return await func(completion_kwargs)

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
        history.add_assistant_message(
            tool_calls=[
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                }
                for tc in native_tool_calls
            ]
        )

        for tool_call in native_tool_calls:
            call_id = tool_call.call_id
            tool_name = tool_call.name
            tool_args = tool_call.args

            content: str = ""
            if tool_name in self.tools:
                try:
                    result = await self.tools[tool_name](**tool_args)

                    if is_module_callback(result):
                        context = PredictContext(module=self, history=history)
                        content = result(context)
                    elif isinstance(result, BaseModel):
                        content = result.model_dump_json()
                    elif not isinstance(result, str):
                        content = json.dumps(result)
                    else:
                        content = result
                except Exception as e:
                    content = f"Error executing tool: {e}"
            else:
                content = f"Error: Tool `{tool_name}` not found."
                available_tools = ", ".join(f"`{tool}`" for tool in self.tools.keys())
                if available_tools:
                    content += f" Available tools are: {available_tools}."
                else:
                    content += " No tools are currently available."

            history.add_tool_result(str(call_id), content)

    def _update_history_with_prediction(self, history: History, prediction: Prediction) -> None:
        """Update history with assistant's prediction.

        Args:
            history: History object to update
            prediction: Prediction from assistant
        """
        output_fields = self.signature.get_output_fields()
        content_parts = []

        for field_name in output_fields:
            if hasattr(prediction, field_name):
                value = getattr(prediction, field_name)
                if value:
                    content_parts.append(f"[[ ## {field_name} ## ]]\n{value}")

        content = "\n".join(content_parts) if content_parts else ""
        history.add_assistant_message(content)

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

            if tool_call.function and tool_call.function.arguments:
                tool_calls[idx]["function"]["arguments"] += tool_call.function.arguments

    def _check_valid_outputs_or_raise(
        self,
        native_tool_calls: list[ToolCall],
        outputs: dict[str, Any],
        completion_text: str,
    ) -> None:
        """
        Check if the tool calls and outputs are valid; raise AdapterParseError if not.
        """
        if not native_tool_calls and outputs.keys() != self.signature.get_output_fields().keys():
            raise AdapterParseError(
                adapter_name=self.adapter.__class__.__name__,
                signature=self.signature,
                lm_response=completion_text,
                parsed_result=outputs,
            )

    @retry(
        retry=retry_if_exception_type(AdapterParseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=3),
    )
    async def _aforward(self, completion_kwargs: dict[str, Any]) -> Prediction:
        """Process non-streaming LLM call with automatic retry on parse errors.

        Retries up to 2 times (3 total attempts) with exponential backoff (0.1-3s)
        when AdapterParseError occurs, giving the LLM multiple chances to format
        the response correctly.

        Args:
            completion_kwargs: Arguments for the completion API call

        Returns:
            Prediction object
        """

        response = await settings.lm.acomplete(**completion_kwargs)

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
        prediction = Prediction(module=self, native_tool_calls=native_tool_calls, **outputs)
        emit_event(prediction)  # If a stream is active, emit the final prediction

        return prediction

    @retry(
        retry=retry_if_exception_type(AdapterParseError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=3),
    )
    async def _astream(self, completion_kwargs: dict[str, Any]) -> Prediction:
        """Process streaming LLM call with automatic retry on parse errors.

        Retries up to 2 times (3 total attempts) with exponential backoff (0.1-3s)
        when AdapterParseError occurs, giving the LLM multiple chances to format
        the response correctly.

        Args:
            completion_kwargs: Arguments for the completion API call

        Returns:
            Prediction object
        """

        try:
            stream: AsyncStream[ChatCompletionChunk] = await settings.lm.acomplete(  # type: ignore[assignment]
                **completion_kwargs
            )

            full_completion: list[str] = []
            tool_calls: dict[int, dict[str, Any]] = {}

            # Use adapter's streaming parser for robust partial JSON parsing
            streaming_parser = self.adapter.create_streaming_parser(self, self.signature)

            reasoning = ThoughtStreamChunk(self, "thought", "", "", is_complete=False)

            async for chunk in stream:
                choice = chunk.choices[0]

                delta = choice.delta.content or ""
                if delta:
                    full_completion.append(delta)
                    await streaming_parser.process_delta(delta)

                if choice.delta.tool_calls:
                    self._process_tool_call_delta(tool_calls, choice.delta.tool_calls)
                elif not reasoning.is_complete:
                    reasoning = await self._process_reasoning_delta(reasoning, choice)

            # Finalize streaming parsing and get outputs
            outputs = await streaming_parser.finalize()
            completion_text = "".join(full_completion)

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

            prediction = Prediction(module=self, native_tool_calls=native_tool_calls, **outputs)
            emit_event(prediction)

            return prediction

        except Exception as exc:
            import traceback

            error_event = type(
                "StreamError",
                (StreamEvent,),
                {"error": str(exc), "traceback": traceback.format_exc(), "module": self},
            )()
            emit_event(error_event)
            raise

    async def _process_reasoning_delta(
        self,
        reasoning: ThoughtStreamChunk,
        choice: Any,
    ) -> ThoughtStreamChunk:
        """Process reasoning delta from choice.

        Args:
            reasoning: Current reasoning chunk state
            choice: Choice object from streaming chunk
            queue: Event queue to emit chunks to

        Returns:
            Updated ThoughtStreamChunk
        """
        # For some reason, AWS Bedrock returns reasoning as content inside <reasoning> tags
        # instead of proper choice.delta.reasoning, so if that happens, we extract it here
        # Unfortunately, we cannot really say it's finished until we get the real content after.
        if match := re.search(
            r"<reasoning>(.*?)</reasoning>", choice.delta.content or "", re.DOTALL
        ):
            thought_chunk = match.group(1)
        else:
            thought_chunk = getattr(choice.delta, "reasoning", None) or ""

        is_complete = choice.finish_reason is not None
        if thought_chunk or is_complete != reasoning.is_complete:
            reasoning = ThoughtStreamChunk(
                self,
                reasoning.field_name,
                thought_chunk,
                reasoning.content + thought_chunk,
                is_complete=is_complete,
            )
            emit_event(reasoning)
        return reasoning


__all__ = ["Predict"]
