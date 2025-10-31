"""ReAct module for reasoning and acting with tools."""

from __future__ import annotations

import json
import logging
import secrets
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from pydantic import Field, create_model

from udspy.callback import with_callbacks
from udspy.confirmation import ConfirmationRequired, respond_to_confirmation
from udspy.decorators import suspendable
from udspy.module.base import Module
from udspy.module.callbacks import ReactContext, is_module_callback
from udspy.module.chain_of_thought import ChainOfThought
from udspy.module.predict import Predict
from udspy.signature import Signature, make_signature
from udspy.streaming import Prediction
from udspy.tool import Tool, Tools
from udspy.utils.formatting import format_tool_exception

# Rebuild Tools model to resolve forward references
Tools.model_rebuild()

logger = logging.getLogger(__name__)


class ToolCallDict(TypedDict):
    """Typed dict for a tool call within an episode."""

    id: str
    name: str
    args: dict[str, Any]


class Episode(TypedDict):
    """Typed dict for a single ReAct episode (thought -> tool calls -> observation)."""

    thought: str
    tool_calls: list[ToolCallDict]
    observation: str


class ReAct(Module):
    """ReAct (Reasoning and Acting) module for tool-using agents.

    ReAct iteratively reasons about the current situation and decides whether
    to call a tool or finish the task. Key features:

    - Iterative reasoning with tool execution
    - Built-in ask_to_user tool for clarification
    - Tool confirmation support for confirmations
    - State saving/restoration for resumption after confirmation requests
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
        enable_ask_to_user: bool = False,
    ):
        """Initialize ReAct module.

        Args:
            signature: Signature defining inputs and outputs, or signature string
            tools: List of tool functions (decorated with @tool) or Tool objects
            max_iters: Maximum number of reasoning iterations (default: 10)
            enable_ask_to_user: Whether to enable ask_to_user tool (default: False)
        """
        if isinstance(signature, str):
            signature = Signature.from_string(signature)

        self.signature = signature
        self.user_signature = signature
        self.max_iters = max_iters
        self.enable_ask_to_user = enable_ask_to_user
        self._context: ReactContext | None = None  # Current execution context

        self.init_module(tools=tools)

    def _init_tools(self) -> None:
        """Initialize tools dictionary with user-provided tools."""
        tool_list = [t if isinstance(t, Tool) else Tool(t) for t in self._tools]
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tool_list if tool.name}
        self._add_builtin_tools()

    def _add_builtin_tools(self) -> None:
        """Add built-in tools: finish and ask_to_user (if enabled)."""
        outputs = ", ".join([f"`{k}`" for k in self.signature.get_output_fields().keys()])

        def finish_tool() -> str:  # pyright: ignore[reportUnusedParameter]
            """Finish tool that accepts and ignores any arguments."""
            return "Task completed"

        self.tools["finish"] = Tool(
            func=finish_tool,
            name="finish",
            description=f"Call this when you have all information needed to produce {outputs}",
        )

        if self.enable_ask_to_user:

            def ask_to_user_impl(question: str) -> str:  # noqa: ARG001
                """Ask the user for clarification."""
                return ""

            self.tools["ask_to_user"] = Tool(
                func=ask_to_user_impl,
                name="ask_to_user",
                description="Ask the user for clarification when needed. Use this when you need more information or when the request is ambiguous.",
                require_confirmation=True,
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
                "To do this, you will interleave next_thought, next_tool_calls in each turn, and also when finishing the task.",
                "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n",
                "When writing next_thought, you reason about the current situation and plan for future steps. If no reasoning is needed, you provide an empty string.",
                "When selecting the next_tool_calls, you may choose one or more of the following tools that will be executed in the same episode:",
            ]
        )

        instr.append(Tools(tools=list(self.tools.values())).format())
        instr.extend(
            [
                "If a tool fails multiple times, call `finish` to complete the task explaining the failure in your final answer.",
                "**CRITICAL: Tool Call Format** You MUST return tool calls in the `next_tool_calls` array field only.",
            ]
        )

        react_input_fields: dict[str, type] = {}
        for name, field_info in self.user_signature.get_input_fields().items():
            react_input_fields[name] = field_info.annotation or str
        react_input_fields["trajectory"] = str

        ToolCallModel = create_model(
            "ToolCall",
            name=(Literal[*self.tools.keys()], ...),
            args=(
                dict[str, Any],
                Field(
                    ...,
                    description="It must be a JSON object matching the tool's arguments schema",
                ),
            ),
        )

        react_output_fields: dict[str, type] = {
            "next_thought": str,
            "next_tool_calls": list[ToolCallModel],  # type: ignore[valid-type]
        }

        return make_signature(
            react_input_fields,
            react_output_fields,
            "\n".join(instr),
        )

    def _build_extract_signature(self) -> type[Signature]:
        """Build extract signature for final answer extraction from trajectory."""
        base_instructions = getattr(self.user_signature, "__doc__", "")

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
            base_instructions or "Extract the final answer from the trajectory",
        )

    def init_module(self, tools: list[Any] | None = None) -> None:
        """Initialize or reinitialize ReAct with new tools.

        This method rebuilds the tools dictionary and regenerates the react
        signature with new tool descriptions. Built-in tools (finish and
        ask_to_user) are automatically preserved.

        Args:
            tools: New tools to initialize with. Can be:
                - Functions decorated with @tool
                - Tool instances
                - None to clear all non-built-in tools

        Example:
            ```python
            from udspy import module_callback

            @module_callback
            def load_specialized_tools(context):
                # Get current non-built-in tools
                current_tools = [
                    t for t in context.module.tools.values()
                    if t.name not in ("finish", "ask_to_user")
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
        for idx, episode in enumerate(trajectory):
            lines.append(f"\n--- Step {idx + 1} ---")
            lines.append(f"Thought: {episode['thought']}")
            lines.append("Tool Calls:")
            for tool_call in episode["tool_calls"]:
                lines.append(f"  {json.dumps(tool_call)}")
            lines.append(f"Observation: {episode['observation']}")

        return "\n".join(lines)

    async def _execute_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_call_id: str,
    ) -> str:
        """Execute a single tool call and return observation.

        Uses self._context for accessing trajectory, input_args, etc.

        Args:
            tool_name: Name of tool to execute
            tool_args: Arguments for the tool
            tool_call_id: Unique ID for this tool call

        Returns:
            Observation string from tool execution

        Raises:
            ConfirmationRequired: When human input is needed
        """
        logger.debug(f"Tool call - name: {tool_name}, args: {tool_args}, id: {tool_call_id}")
        tool = None
        try:
            tool = self.tools[tool_name]
            result = await tool.acall(**tool_args)

            if is_module_callback(result):
                # Pass module's context to callback
                if self._context is None:
                    raise RuntimeError("Module callback called outside execution context")
                observation = result(self._context)
            else:
                observation = str(result)

            return f"{tool_call_id}. {observation}"
        except ConfirmationRequired as e:
            # Store context for resumption
            if self._context is not None:
                e.context = {
                    "trajectory": self._context.trajectory.copy(),
                    "input_args": self._context.input_args.copy(),
                    "stream": self._context.stream,
                }
            if e.tool_call and tool_call_id:
                e.tool_call.call_id = tool_call_id
            raise
        except Exception as e:
            parts = [
                f"{tool_call_id}.",
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
        pending_episode: Episode | None = None,
    ) -> bool:
        """Execute a single ReAct iteration (create one episode).

        Uses self._context for trajectory and input_args.

        Args:
            stream: Whether to stream sub-module execution
            pending_episode: Optional partial episode to complete (for resumption after confirmation)

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

        # If we have a pending episode, complete it by executing remaining tool calls
        if pending_episode:
            observations = []
            should_stop = False

            for tool_call in pending_episode["tool_calls"]:
                observation = await self._execute_tool_call(
                    tool_call["name"],
                    tool_call["args"],
                    tool_call["id"],
                )
                observations.append(observation)
                should_stop = tool_call["name"] == "finish"

            pending_episode["observation"] = "\n".join(observations)
            trajectory.append(pending_episode)
            return should_stop

        # Normal flow: get next thought and tool calls from LLM
        formatted_trajectory = self._format_trajectory(trajectory)
        pred = await self.react_module.aexecute(
            stream=stream,
            **input_args,
            trajectory=formatted_trajectory,
        )

        thought = pred.get("next_thought", "").strip()

        next_tool_calls = pred.get("next_tool_calls", [])
        if isinstance(next_tool_calls, dict) and "items" in next_tool_calls:
            next_tool_calls = next_tool_calls["items"]

        if not isinstance(next_tool_calls, list):
            episode: Episode = {
                "thought": thought,
                "tool_calls": [],
                "observation": f"Error: Malformed next_tool_calls (expected list, got {type(next_tool_calls).__name__})",
            }
            trajectory.append(episode)
            return False

        # Build tool calls list with IDs
        tool_calls_with_ids: list[ToolCallDict] = []
        for tool_call in next_tool_calls:
            tool_calls_with_ids.append(
                {
                    "id": f"call_{secrets.token_hex(8)}",
                    "name": tool_call["name"],
                    "args": tool_call["args"],
                }
            )

        # Execute all tool calls and collect observations
        observations = []
        should_stop = False
        for tool_call in tool_calls_with_ids:
            observation = await self._execute_tool_call(
                tool_call["name"],
                tool_call["args"],
                tool_call["id"],
            )
            observations.append(observation)
            should_stop = tool_call["name"] == "finish"

        # Create and append completed episode
        episode = {
            "thought": thought,
            "tool_calls": tool_calls_with_ids,
            "observation": "\n".join(observations),
        }
        trajectory.append(episode)

        return should_stop

    @suspendable
    @with_callbacks
    async def aexecute(
        self,
        *,
        stream: bool = False,
        _trajectory: list[Episode] | None = None,
        _pending_episode: Episode | None = None,
        **input_args: Any,
    ) -> Prediction:
        """Execute ReAct loop.

        Args:
            stream: Passed to sub-modules
            _trajectory: Internal - restored trajectory for resumption (list of completed episodes)
            _pending_episode: Internal - partial episode to complete for resumption
            history: History object for streaming (not used currently)
            **input_args: Input values matching signature's input fields

        Returns:
            Prediction with trajectory and output fields

        Raises:
            ConfirmationRequired: When human input is needed
        """
        max_iters = input_args.pop("max_iters", self.max_iters)
        trajectory: list[Episode] = _trajectory if _trajectory is not None else []

        # Set up React context for this execution
        self._context = ReactContext(
            module=self,
            trajectory=trajectory,
            input_args=input_args,
            stream=stream,
        )

        try:
            # If we have a pending episode, complete it first
            if _pending_episode:
                try:
                    should_stop = await self._execute_iteration(
                        stream=stream,
                        pending_episode=_pending_episode,
                    )
                    if should_stop:
                        formatted_trajectory = self._format_trajectory(trajectory)
                        extract = await self.extract_module.aexecute(
                            stream=stream,
                            **input_args,
                            trajectory=formatted_trajectory,
                        )

                        result_dict: dict[str, Any] = {}
                        for field_name in self.signature.get_output_fields():
                            if hasattr(extract, field_name):
                                result_dict[field_name] = getattr(extract, field_name)

                        return Prediction(
                            module=self,
                            is_final=True,
                            trajectory=trajectory,
                            **result_dict,
                        )
                except ValueError as e:
                    logger.warning(f"Agent failed: {e}")
                    error_episode: Episode = {
                        "thought": "",
                        "tool_calls": [],
                        "observation": f"Error: {e}",
                    }
                    trajectory.append(error_episode)

            # Continue with normal iteration loop
            while len(trajectory) < max_iters:
                try:
                    should_stop = await self._execute_iteration(
                        stream=stream,
                    )
                    if should_stop:
                        break

                except ValueError as e:
                    logger.warning(f"Agent failed to select valid tool: {e}")
                    error_episode = {
                        "thought": "",
                        "tool_calls": [],
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

            result_dict = {}
            for field_name in self.signature.get_output_fields():
                if hasattr(extract, field_name):
                    result_dict[field_name] = getattr(extract, field_name)

            prediction = Prediction(
                module=self, is_final=True, trajectory=trajectory, **result_dict
            )
            return prediction
        finally:
            # Clean up context
            self._context = None

    async def aresume(
        self,
        user_response: str,
        saved_state: Any,
    ) -> Prediction:
        """Async resume execution after user input.

        This method only resolves the pending tool call based on user response,
        then delegates to aexecute to continue the normal flow with the stream
        parameter that was originally passed.

        Args:
            user_response: The user's response. Can be:
                - "yes"/"y" to confirm tool execution with original args
                - "no"/"n" to reject and continue
                - JSON dict string to execute tool with modified args
                - Any other text is treated as user feedback for LLM to re-reason

        Returns:
            Final prediction after resuming

        Raises:
            ConfirmationRequired: If another human input is needed
        """
        trajectory: list[Episode] = saved_state.context.get("trajectory", []).copy()
        input_args: dict[str, Any] = saved_state.context.get("input_args", {}).copy()
        stream: bool = saved_state.context.get("stream", False)

        user_response_lower = user_response.lower().strip()
        pending_episode: Episode | None = None

        if user_response_lower in ("yes", "y"):
            # Approve and execute with original args
            if saved_state.confirmation_id:
                respond_to_confirmation(
                    saved_state.confirmation_id, approved=True, status="approved"
                )
            if saved_state.tool_call:
                # Create pending episode with the confirmed tool call
                pending_episode = {
                    "thought": "",  # No new thought, continuing from confirmation
                    "tool_calls": [
                        {
                            "id": saved_state.tool_call.call_id or f"call_{secrets.token_hex(8)}",
                            "name": saved_state.tool_call.name,
                            "args": saved_state.tool_call.args.copy(),
                        }
                    ],
                    "observation": "",  # Will be filled by _execute_iteration
                }
        elif user_response_lower in ("no", "n"):
            # Reject and add feedback episode
            if saved_state.confirmation_id:
                respond_to_confirmation(
                    saved_state.confirmation_id, approved=False, status="rejected"
                )
            feedback_episode: Episode = {
                "thought": "",
                "tool_calls": [],
                "observation": "User rejected the operation",
            }
            trajectory.append(feedback_episode)
        else:
            # Try to parse as JSON for edited args, otherwise treat as feedback
            try:
                modified_args = json.loads(user_response)
                if isinstance(modified_args, dict):
                    # Execute with modified args
                    if saved_state.confirmation_id:
                        respond_to_confirmation(
                            saved_state.confirmation_id,
                            approved=True,
                            data=modified_args,
                            status="edited",
                        )
                    if saved_state.tool_call:
                        pending_episode = {
                            "thought": "",
                            "tool_calls": [
                                {
                                    "id": saved_state.tool_call.call_id
                                    or f"call_{secrets.token_hex(8)}",
                                    "name": saved_state.tool_call.name,
                                    "args": modified_args,
                                }
                            ],
                            "observation": "",
                        }
                else:
                    # Non-dict JSON, treat as feedback
                    if saved_state.confirmation_id:
                        respond_to_confirmation(
                            saved_state.confirmation_id,
                            approved=False,
                            status="feedback",
                        )
                    feedback_episode = {
                        "thought": "",
                        "tool_calls": [],
                        "observation": f"User feedback: {user_response}",
                    }
                    trajectory.append(feedback_episode)
            except json.JSONDecodeError:
                # Not JSON, treat as natural language feedback
                if saved_state.confirmation_id:
                    respond_to_confirmation(
                        saved_state.confirmation_id, approved=False, status="feedback"
                    )
                feedback_episode = {
                    "thought": "",
                    "tool_calls": [],
                    "observation": f"User feedback: {user_response}",
                }
                trajectory.append(feedback_episode)

        # Continue execution with prepared state, respecting original stream parameter
        return await self.aexecute(
            stream=stream,
            _trajectory=trajectory,
            _pending_episode=pending_episode,
            **input_args,
        )
