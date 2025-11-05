"""ReAct module for reasoning and acting with tools."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from udspy.callback import with_callbacks
from udspy.confirmation import ConfirmationRequired
from udspy.history import History
from udspy.module.base import Module
from udspy.module.callbacks import ReactContext, is_module_callback
from udspy.module.chain_of_thought import ChainOfThought
from udspy.module.predict import Predict
from udspy.signature import Signature, make_signature
from udspy.streaming import Prediction, emit_event
from udspy.tool import Tool, Tools
from udspy.utils.async_support import execute_function_async
from udspy.utils.formatting import format_tool_exception

# Rebuild Tools model to resolve forward references
Tools.model_rebuild()

logger = logging.getLogger(__name__)


class Episode(TypedDict):
    """Typed dict for a single ReAct episode (thought -> tool calls -> observation)."""

    thought: str
    tool_name: str | None
    tool_args: dict[str, Any] | None
    observation: str


class ReAct(Module):
    """ReAct (Reasoning and Acting) module for tool-using agents.

    ReAct iteratively reasons about the current situation and decides whether
    to call a tool or finish the task. Key features:

    - Iterative reasoning with tool execution
    - Tool confirmation support for sensitive operations
    - Real-time streaming of reasoning and tool usage

    Example (Basic Usage):
        ```python
        from udspy import ReAct, Signature, InputField, OutputField, tool
        from pydantic import Field

        @tool(name="search", description="Search for information")
        def search(query: str = Field(...)) -> str:
            return f"Results for: {query}"

        class QA(Signature):
            '''Answer questions using available tools.'''
            question: str = InputField()
            answer: str = OutputField()

        react = ReAct(QA, tools=[search])
        result = react(question="What is the weather in Tokyo?")
        ```

    Example (Streaming):
        ```python
        # Stream the agent's reasoning process in real-time
        async for event in react.astream(question="What is Python?"):
            if isinstance(event, OutputStreamChunk):
                print(event.delta, end="", flush=True)
            elif isinstance(event, Prediction):
                print(f"Answer: {event.answer}")
        ```

        See examples/react_streaming.py for a complete streaming example.

    Example (Tools with Confirmation):
        ```python
        from udspy import ConfirmationRequired, ConfirmationRejected

        @tool(name="delete_file", require_confirmation=True)
        def delete_file(path: str = Field(...)) -> str:
            return f"Deleted {path}"

        react = ReAct(QA, tools=[delete_file])

        try:
            result = await react.aforward(question="Delete /tmp/test.txt")
        except ConfirmationRequired as e:
            # User is asked for confirmation
            print(f"Confirm: {e.question}")
            # Approve: respond_to_confirmation(e.confirmation_id, approved=True)
            # Reject: respond_to_confirmation(e.confirmation_id, approved=False, status="rejected")
            result = await react.aresume("yes", e)
        ```
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        tools: list[Callable | Tool],
        *,
        max_iters: int = 10,
    ):
        """Initialize ReAct module.

        Args:
            signature: Signature defining inputs and outputs, or signature string
            tools: List of tool functions (decorated with @tool) or Tool objects
            max_iters: Maximum number of reasoning iterations (default: 10)
        """
        if isinstance(signature, str):
            signature = Signature.from_string(signature)

        self.signature = signature
        self.user_signature = signature
        self.max_iters = max_iters
        self._context: ReactContext | None = None  # Current execution context

        self.init_module(tools=tools)

    def _init_tools(self) -> None:
        """Initialize tools dictionary with user-provided tools."""
        tool_list = [t if isinstance(t, Tool) else Tool(t) for t in self._tools]
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tool_list if tool.name}
        self._add_builtin_tools()

    def _add_builtin_tools(self) -> None:
        """Add built-in finish tool."""
        outputs = ", ".join([f"`{k}`" for k in self.signature.get_output_fields().keys()])

        def finish_tool() -> str:  # pyright: ignore[reportUnusedParameter]
            """Finish tool that accepts and ignores any arguments."""
            return "Task completed"

        self.tools["finish"] = Tool(
            func=finish_tool,
            name="finish",
            description=f"Call this when you have all information needed to produce {outputs}",
        )

    def _rebuild_signatures(self) -> None:
        """Rebuild react and extract signatures with current tools.

        This method reconstructs the signatures used by the ReAct module,
        incorporating the current set of tools. It's called during initialization
        and when tools are dynamically updated via init_module().
        """
        self.react_signature = self._build_react_signature()
        self.extract_signature = self._build_extract_signature()
        self.react_module = Predict(self.react_signature)
        self.extract_module = ChainOfThought(self.extract_signature)

    def _build_react_signature(self) -> type[Signature]:
        """Build ReAct signature with tool descriptions in instructions."""
        inputs = ", ".join([f"`{k}`" for k in self.user_signature.get_input_fields().keys()])
        outputs = ", ".join([f"`{k}`" for k in self.user_signature.get_output_fields().keys()])

        base_instructions = getattr(self.user_signature, "__doc__", "")
        instr = [f"{base_instructions}\n"] if base_instructions else []

        instr.extend(
            [
                f"You are an Agent. In each episode, you will be given the fields {inputs} as input. And you can see your past trajectory so far.",
                f"Your goal is to use one or more of the supplied tools to collect any necessary information for producing {outputs}.\n",
                "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.",
                "After each step, you receive a resulting observation, which gets appended to your trajectory.\n",
                "When writing next_thought, you may reason about the current situation and plan for future steps.",
                "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n",
            ]
        )

        instr.append(Tools(tools=list(self.tools.values())).format())
        instr.extend(
            [
                "IMPORTANT: You must respond with a JSON object in your message content containing the fields: "
                '{"next_thought": "...", "next_tool_name": "...", "next_tool_args": {...}}.',
                "NEVER use function calling or tool calling syntax - return the JSON as plain text in your response.",
            ]
        )

        react_input_fields: dict[str, type] = {
            "trajectory": str,
            "messages": list[dict[str, Any]],
        }
        for name, field_info in self.user_signature.get_input_fields().items():
            react_input_fields[name] = field_info.annotation or str

        react_output_fields: dict[str, type] = {
            "next_thought": str,
            "next_tool_name": Literal[*self.tools.keys()],  # type: ignore[dict-item]
            "next_tool_args": dict[str, Any],
        }

        return make_signature(
            react_input_fields,
            react_output_fields,
            "\n".join(instr),
        )

    def _build_extract_signature(self) -> type[Signature]:
        """Build extract signature for final answer extraction from trajectory."""
        extract_input_fields: dict[str, type] = {}
        extract_output_fields: dict[str, type] = {}

        for name, field_info in self.user_signature.get_input_fields().items():
            extract_input_fields[name] = field_info.annotation or str

        for name, field_info in self.user_signature.get_output_fields().items():
            extract_output_fields[name] = field_info.annotation or str

        extract_input_fields["trajectory"] = str

        return make_signature(
            extract_input_fields,
            extract_output_fields,
            "Extract the final answer from the trajectory",
        )

    def init_module(self, tools: list[Any] | None = None) -> None:
        """Initialize or reinitialize ReAct with new tools.

        This method rebuilds the tools dictionary and regenerates the react signature
        with new tool descriptions. Built-in tools are automatically preserved.

        Args:
            tools: New tools to initialize with. Can be:
                - Functions decorated with @tool
                - Tool instances
                - None to clear all non-built-in tools

        Example:
            ```python from udspy import module_callback

            @module_callback
            def load_specialized_tools(context):
                # Get current non-built-in tools
                current_tools = [
                    t for t in context.module.tools.values()
                    if t.name not in builtin_tool_names
                ]

                # Add new tools
                new_tools = [weather_tool, calendar_tool]

                # Reinitialize with all tools
                context.module.init_module(tools=current_tools + new_tools)

                return f"Added {len(new_tools)} specialized tools"
            ```
        """

        self._tools = tools or []
        self._init_tools()
        self._rebuild_signatures()

    def _format_trajectory(self, trajectory: list[Episode]) -> str:
        """Format trajectory as a string for the LLM.

        Args:
            trajectory: List of episodes

        Returns:
            Formatted string representation
        """
        if not trajectory:
            return "No actions taken yet."

        lines = []
        for step, episode in enumerate(trajectory, start=1):
            lines.append(json.dumps({"step": step, **episode}))

        return "\n".join(lines)

    async def _execute_tool_call(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a single tool call and return observation.

        Uses self._context for accessing trajectory, input_args, etc.

        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments for the tool

        Returns:
            Observation string from tool execution

        Raises:
            ConfirmationRequired: When human input is needed
        """
        logger.debug(f"Tool call - name: {tool_name}, args: {tool_args}")
        tool = None
        try:
            tool = self.tools[tool_name]
            result = await tool.acall(**tool_args)

            if is_module_callback(result):
                # Pass module's context to callback
                if self._context is None:
                    raise RuntimeError("Module callback called outside execution context")
                observation = await execute_function_async(result, {"context": self._context})
            else:
                observation = str(result)

            return observation
        except ConfirmationRequired as e:
            # Store context for resumption
            if self._context is not None:
                e.context = {
                    "trajectory": self._context.trajectory.copy(),
                    "input_args": self._context.input_args.copy(),
                    "stream": self._context.stream,
                }
            raise
        except Exception as e:
            parts = [
                f"Traceback '{tool_name}': {format_tool_exception(e)}.",
            ]
            if tool is not None:
                parts.append(f"Expected tool args schema: {tool.parameters}.")
            logger.warning(f"Tool execution failed: {e}")
            return " ".join(parts)

    async def _execute_iteration(
        self,
        *,
        stream: bool = False,
    ) -> bool:
        """
        Execute a single ReAct iteration (create one episode).
        Uses self._context for trajectory and input_args.

        Args:
            stream: Whether to stream sub-module execution

        Returns:
            should_stop: Whether to stop the ReAct loop

        Raises:
            ConfirmationRequired: When human input is needed
        """
        # Get context from instance
        if self._context is None:
            raise RuntimeError("_execute_iteration called outside execution context")

        trajectory = self._context.trajectory
        input_args = self._context.input_args
        history = self._context.history

        # Normal flow: get next thought and tool calls from LLM
        formatted_trajectory = self._format_trajectory(trajectory)
        pred = await self.react_module.aexecute(
            stream=stream,
            **input_args,
            messages=history.messages if history else [],
            trajectory=formatted_trajectory,
        )

        thought = pred.get("next_thought", "").strip()
        tool_name = pred.get("next_tool_name", None)
        if tool_name not in self.tools:
            raise ValueError(
                "Invalid tool name selected by agent. Available tools: , ".join(
                    f"`{name}`" for name in self.tools.keys()
                )
            )

        tool_args = pred.get("next_tool_args", None)
        observation = await self._execute_tool_call(tool_name, tool_args)

        episode: Episode = {
            "thought": thought,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "observation": observation,
        }
        trajectory.append(episode)

        should_stop = tool_name == "finish"
        return should_stop

    @with_callbacks
    async def aexecute(
        self,
        *,
        stream: bool = False,
        _trajectory: list[Episode] | None = None,
        history: History | None = None,
        **input_args: Any,
    ) -> Prediction:
        """Execute ReAct loop.

        Args:
            stream: Passed to sub-modules
            _trajectory: Internal - restored trajectory for resumption (list of completed episodes)
            history: History object for streaming (not used currently)
            **input_args: Input values matching signature's input fields

        Returns:
            Prediction with trajectory and output fields

        Raises:
            ConfirmationRequired: When human input is needed
        """
        max_iters = input_args.pop("max_iters", self.max_iters)
        trajectory: list[Episode] = _trajectory if _trajectory is not None else []
        if history is None:
            history = History()

        # Set up React context for this execution
        self._context = ReactContext(
            module=self,
            trajectory=trajectory,
            input_args=input_args,
            history=history,
            stream=stream,
        )

        try:
            # Continue with normal iteration loop
            while len(trajectory) < max_iters:
                try:
                    should_stop = await self._execute_iteration(stream=stream)
                    if should_stop:
                        break

                except ValueError as e:
                    logger.warning(f"Agent failed to select valid tool: {e}")
                    error_episode: Episode = {
                        "thought": "",
                        "tool_name": None,
                        "tool_args": None,
                        "observation": f"Error: {e}",
                    }
                    trajectory.append(error_episode)
                    break

            formatted_trajectory = self._format_trajectory(trajectory)
            extract = await self.extract_module.aexecute(
                stream=stream,
                **input_args,
                trajectory=formatted_trajectory,
            )
            result_dict = {
                key: value
                for key, value in extract.items()
                if key in self.signature.get_output_fields()
            }
            history.add_assistant_message(json.dumps(result_dict))

            prediction = Prediction(
                **result_dict,
                reasoning=extract["reasoning"],
                trajectory=trajectory,
                module=self,
            )
            emit_event(prediction)
            return prediction
        finally:
            # Clean up context
            self._context = None
