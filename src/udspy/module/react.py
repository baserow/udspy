"""ReAct module for reasoning and acting with tools."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, Literal, TypedDict

from udspy.callback import with_callbacks
from udspy.confirmation import ConfirmationRejected, ConfirmationRequired
from udspy.decorators import suspendable
from udspy.history import History
from udspy.module.base import Module
from udspy.module.callbacks import ReactContext, is_module_callback
from udspy.module.chain_of_thought import ChainOfThought
from udspy.module.predict import Predict
from udspy.signature import Signature, make_signature
from udspy.streaming import Prediction, emit_event
from udspy.tool import Tool, ToolCall, Tools
from udspy.utils.async_support import execute_function_async
from udspy.utils.formatting import format_tool_exception, format_validation_error

# Rebuild Tools model to resolve forward references
Tools.model_rebuild()

logger = logging.getLogger(__name__)


class PlanItem(TypedDict):
    """A single item in the agent's plan."""

    task: str
    status: Literal["todo", "done"]
    done_at_step: int | None


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
        enable_ask_to_user: bool = False,
        **kwargs: Any,
    ):
        """Initialize ReAct module.

        Args:
            signature: Signature defining inputs and outputs, or signature string
            tools: List of tool functions (decorated with @tool) or Tool objects
            max_iters: Maximum number of reasoning iterations (default: 10)
            enable_ask_to_user: Whether to add an ask_to_user built-in tool
                that lets the agent ask the user clarifying questions
        """
        if isinstance(signature, str):
            signature = Signature.from_string(signature)

        self.signature = signature
        self.user_signature = signature
        self.max_iters = max_iters
        self._enable_ask_to_user = enable_ask_to_user
        self._kwargs = kwargs
        self._context: ReactContext | None = None  # Current execution context

        self.init_module(tools=tools)

    def _init_tools(self) -> None:
        """Initialize tools dictionary with user-provided tools."""
        tool_list = [t if isinstance(t, Tool) else Tool(t) for t in self._tools]
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tool_list if tool.name}
        self._add_builtin_tools()

    def _add_builtin_tools(self) -> None:
        """Add built-in tools (finish, and optionally ask_to_user)."""
        outputs = ", ".join([f"`{k}`" for k in self.signature.get_output_fields().keys()])

        def finish_tool() -> str:  # pyright: ignore[reportUnusedParameter]
            """Finish tool that accepts and ignores any arguments."""
            return "Task completed"

        self.tools["finish"] = Tool(
            func=finish_tool,
            name="finish",
            description=f"Call this when you have all information needed to produce {outputs}",
        )

        if self._enable_ask_to_user:

            def ask_to_user_func(question: str) -> str:
                """Ask the user a clarifying question."""
                raise ConfirmationRequired(
                    question=question,
                    tool_call=ToolCall(name="ask_to_user", args={"question": question}),
                )

            self.tools["ask_to_user"] = Tool(
                func=ask_to_user_func,
                name="ask_to_user",
                description="Ask the user a clarifying question when the request is ambiguous",
            )

    def _rebuild_signatures(self) -> None:
        """Rebuild react and extract signatures with current tools.

        This method reconstructs the signatures used by the ReAct module,
        incorporating the current set of tools. It's called during initialization
        and when tools are dynamically updated via init_module().
        """
        self.react_signature = self._build_react_signature()
        self.extract_signature = self._build_extract_signature()
        self.react_module = Predict(self.react_signature, **self._kwargs)
        self.extract_module = ChainOfThought(self.extract_signature, **self._kwargs)

    def _build_react_signature(self) -> type[Signature]:
        """Build ReAct signature with tool descriptions in instructions."""
        inputs = ", ".join([f"`{k}`" for k in self.user_signature.get_input_fields().keys()])
        outputs = ", ".join([f"`{k}`" for k in self.user_signature.get_output_fields().keys()])

        base_instructions = getattr(self.user_signature, "__doc__", "")
        instr = [f"{base_instructions}\n"] if base_instructions else []

        instr.extend(
            [
                f"You are an Agent. Given {inputs} and your trajectory, use tools to produce {outputs}.",
                "You respond with next_thought, plan_updates, next_tool_name, and next_tool_args each turn.\n",
                "next_thought: reason about the user request vs what the trajectory and plan show is already done. "
                "If a tool succeeded, mark that plan item done and move on. "
                "If a tool failed, retry ONCE only if the fix is obvious from the error. "
                "After two similar failures, call `finish` and report the issue. "
                "NEVER repeat a tool call with the same or similar arguments that already succeeded.\n",
                "plan_updates: a list of updates to the plan. "
                'Use {"add": "task description"} to add a new todo item. '
                'Use {"done": <index>} to mark the item at that index as done. '
                "On your first turn, add all the steps needed. On subsequent turns, mark completed items done and add new ones if needed. "
                "When all items are done, call `finish`.\n",
                "The current plan is provided as input. Items marked [done at step N] are completed — do not redo them.\n",
                "When selecting next_tool_name and next_tool_args, the tool must be one of:\n",
            ]
        )

        instr.append(Tools(tools=list(self.tools.values())).format())
        instr.extend(
            [
                "IMPORTANT: You must respond with a JSON object in your message content containing the fields: "
                '{"next_thought": "...", "plan_updates": [{"add": "..."}, {"done": 0}], "next_tool_name": "...", "next_tool_args": {...}}.',
                "NEVER use function calling or tool calling syntax - return the JSON as plain text in your response.",
            ]
        )

        react_input_fields: dict[str, type] = {
            "trajectory": str,
            "plan": str,
        }
        for name, field_info in self.user_signature.get_input_fields().items():
            react_input_fields[name] = field_info.annotation or str

        react_output_fields: dict[str, type] = {
            "next_thought": str,
            "plan_updates": list[dict[str, Any]],
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

    @staticmethod
    def _format_plan(plan: list[PlanItem]) -> str:
        """Format the plan as a string for the LLM input."""
        if not plan:
            return "No plan yet. Use plan_updates to create one."

        lines = []
        for i, item in enumerate(plan):
            if item["status"] == "done":
                lines.append(f"[{i}] [done at step {item['done_at_step']}] {item['task']}")
            else:
                lines.append(f"[{i}] [todo] {item['task']}")
        return "\n".join(lines)

    @staticmethod
    def _apply_plan_updates(
        plan: list[PlanItem], updates: list[dict[str, Any]], step_number: int
    ) -> None:
        """Apply LLM-proposed plan updates to the canonical plan in place."""
        for update in updates:
            if "add" in update and isinstance(update["add"], str):
                plan.append({"task": update["add"], "status": "todo", "done_at_step": None})
            elif "done" in update:
                idx = update["done"]
                if isinstance(idx, int) and 0 <= idx < len(plan) and plan[idx]["status"] == "todo":
                    plan[idx]["status"] = "done"
                    plan[idx]["done_at_step"] = step_number

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
                    "plan": [item.copy() for item in self._context.plan],
                    "stream": self._context.stream,
                }
            raise
        except Exception as e:
            from pydantic import ValidationError

            logger.warning(f"Tool execution failed: {e}")

            if isinstance(e, (ValidationError, TypeError)):
                observation = format_validation_error(tool_name, e, tool)
            else:
                observation = f"Runtime error in '{tool_name}': {format_tool_exception(e)}"

            # Count consecutive failures for this tool to help the LLM
            # decide when to stop retrying.
            if self._context is not None:
                consecutive = 0
                for ep in reversed(self._context.trajectory):
                    if ep["tool_name"] == tool_name and ep["observation"].startswith(
                        ("Runtime error", "Validation error")
                    ):
                        consecutive += 1
                    else:
                        break
                if consecutive >= 1:
                    observation += (
                        f"\n\nWARNING: '{tool_name}' has now failed "
                        f"{consecutive + 1} times in a row. "
                        "Consider a different approach or call `finish` "
                        "to report the issue."
                    )

            return observation

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

        # Normal flow: get next thought and tool calls from LLM
        plan = self._context.plan
        formatted_trajectory = self._format_trajectory(trajectory)
        formatted_plan = self._format_plan(plan)
        pred = await self.react_module.aexecute(
            stream=stream,
            **input_args,
            trajectory=formatted_trajectory,
            plan=formatted_plan,
        )

        thought = pred.get("next_thought", "").strip()
        plan_updates = pred.get("plan_updates", [])
        tool_name = pred.get("next_tool_name", None)
        if tool_name not in self.tools:
            raise ValueError(
                "Invalid tool name selected by agent. Available tools: , ".join(
                    f"`{name}`" for name in self.tools.keys()
                )
            )

        tool_args = pred.get("next_tool_args", None)
        try:
            observation = await self._execute_tool_call(tool_name, tool_args)
        except ConfirmationRequired as e:
            # Save the current iteration state so aresume() can complete it
            e.context["pending_thought"] = thought
            e.context["pending_plan_updates"] = plan_updates
            raise

        # Apply plan updates after tool execution (step number = next trajectory index)
        step_number = len(trajectory) + 1
        self._apply_plan_updates(plan, plan_updates, step_number)

        episode: Episode = {
            "thought": thought,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "observation": observation,
        }
        trajectory.append(episode)

        should_stop = tool_name == "finish"

        # System guard: force stop if all plan items are done
        if not should_stop and plan and all(item["status"] == "done" for item in plan):
            logger.info("All plan items done — forcing stop")
            should_stop = True

        return should_stop

    @suspendable
    @with_callbacks
    async def aexecute(
        self,
        *,
        stream: bool = False,
        _trajectory: list[Episode] | None = None,
        _plan: list[PlanItem] | None = None,
        history: History | None = None,
        **input_args: Any,
    ) -> Prediction:
        """Execute ReAct loop.

        Args:
            stream: Passed to sub-modules
            _trajectory: Internal - restored trajectory for resumption (list of completed episodes)
            _plan: Internal - restored plan for resumption
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
            plan=_plan,
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
                plan=self._context.plan,
                module=self,
            )
            emit_event(prediction)
            return prediction
        finally:
            # Clean up context
            self._context = None

    async def aresume(self, user_response: str, saved_state: Any) -> Prediction:
        """Resume ReAct execution after a ConfirmationRequired exception.

        Completes the interrupted iteration using the saved state and user's
        response, then continues the ReAct loop.

        Args:
            user_response: The user's response (e.g. "yes", "no", or JSON edits)
            saved_state: The ConfirmationRequired exception with saved context

        Returns:
            Final Prediction after completing the ReAct loop

        Raises:
            ConfirmationRejected: If user rejected the operation
            ConfirmationRequired: If another confirmation is needed during execution
        """
        from udspy.confirmation import respond_to_confirmation

        exc = saved_state
        ctx = exc.context

        trajectory = ctx.get("trajectory", [])
        input_args = ctx.get("input_args", {})
        plan = ctx.get("plan", [])
        stream = ctx.get("stream", False)

        tool_name = exc.tool_call.name if exc.tool_call else None
        tool_args = exc.tool_call.args if exc.tool_call else {}

        # Handle ask_to_user: user's response becomes the observation directly
        if tool_name == "ask_to_user":
            observation = f"User response: {user_response}"
        elif tool_name is not None:
            # Handle tool confirmation
            response_lower = user_response.lower().strip()

            if response_lower in ("no", "n"):
                raise ConfirmationRejected(
                    message=f"User rejected execution of {tool_name}",
                    confirmation_id=exc.confirmation_id,
                    tool_call=exc.tool_call,
                )

            # Set approval in confirmation context before re-executing the tool
            if response_lower.startswith("{"):
                try:
                    edited_args = json.loads(user_response)
                    respond_to_confirmation(
                        exc.confirmation_id,
                        approved=True,
                        data=edited_args,
                        status="edited",
                    )
                except json.JSONDecodeError:
                    respond_to_confirmation(exc.confirmation_id, approved=True)
            else:
                respond_to_confirmation(exc.confirmation_id, approved=True)

            # Re-execute the tool (now with approval set in confirmation context)
            self._context = ReactContext(
                module=self,
                trajectory=trajectory,
                input_args=input_args,
                plan=plan,
                stream=stream,
            )
            try:
                observation = await self._execute_tool_call(tool_name, tool_args)
            finally:
                self._context = None
        else:
            observation = f"Unknown tool confirmation for: {exc.question}"

        # Complete the pending episode
        pending_thought = ctx.get("pending_thought", "")
        pending_plan_updates = ctx.get("pending_plan_updates", [])

        step_number = len(trajectory) + 1
        self._apply_plan_updates(plan, pending_plan_updates, step_number)

        episode: Episode = {
            "thought": pending_thought,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "observation": observation,
        }
        trajectory.append(episode)

        # Continue the ReAct loop
        return await self.aexecute(
            stream=stream,
            _trajectory=trajectory,
            _plan=plan,
            **input_args,
        )
