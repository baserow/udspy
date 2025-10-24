"""ReAct module for reasoning and acting with tools."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from udspy.module.base import Module
from udspy.module.chain_of_thought import ChainOfThought
from udspy.module.predict import Predict
from udspy.signature import Signature, make_signature
from udspy.streaming import Prediction

if TYPE_CHECKING:
    from udspy.tool import Tool

logger = logging.getLogger(__name__)


class HumanInTheLoopRequired(Exception):
    """Raised when human input is needed to proceed.

    This exception can be raised by tools or the ReAct module to pause execution
    and request human input. It includes context for resuming execution later.

    Common use cases:
    - Tool requesting confirmation before destructive operations
    - Tool asking for clarification or additional information
    - Agent asking user to resolve ambiguity
    """

    def __init__(
        self,
        question: str,
        *,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_args: dict[str, Any] | None = None,
        trajectory: dict[str, Any] | None = None,
        iteration: int | None = None,
        input_args: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize with question and optional context.

        Args:
            question: Question to ask the human
            tool_name: Name of the tool that raised this (if any)
            tool_call_id: Tool call ID for tracking (if any)
            tool_args: Arguments passed to the tool (if any)
            trajectory: Current trajectory state (set by ReAct)
            iteration: Current iteration number (set by ReAct)
            input_args: Original input arguments (set by ReAct)
            context: Additional context dictionary
        """
        self.question = question
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self.tool_args = tool_args or {}
        self.trajectory = trajectory or {}
        self.iteration = iteration
        self.input_args = input_args or {}
        self.context = context or {}
        super().__init__(f"Human input required: {question}")


# Backwards compatibility alias
UserInputRequired = HumanInTheLoopRequired


class ReAct(Module):
    """ReAct (Reasoning and Acting) module for tool-using agents.

    ReAct iteratively reasons about the current situation and decides whether
    to call a tool or finish the task. This implementation adds:

    - ask_to_user tool for clarification (usable once at start or after failures)
    - Tool confirmation support via ask_for_confirmation flag
    - State saving/restoration for user interaction
    - Trajectory truncation for long contexts

    Example:
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
    """

    def __init__(
        self,
        signature: type[Signature] | str,
        tools: list[Callable | Tool],
        *,
        max_iters: int = 10,
        max_failures: int = 3,
        enable_ask_to_user: bool = True,
    ):
        """Initialize ReAct module.

        Args:
            signature: Signature defining inputs and outputs, or signature string
            tools: List of tool functions (decorated with @tool) or Tool objects
            max_iters: Maximum number of reasoning iterations (default: 10)
            max_failures: Number of consecutive failures before allowing ask_to_user (default: 3)
            enable_ask_to_user: Whether to enable ask_to_user tool (default: True)
        """
        from udspy.tool import Tool

        super().__init__()

        # Handle string signatures (simple format: "input1, input2 -> output1, output2")
        if isinstance(signature, str):
            # Parse simple signature format
            parts = signature.split("->")
            if len(parts) != 2:
                raise ValueError(
                    "String signature must be in format 'input1, input2 -> output1, output2'"
                )

            input_names = [name.strip() for name in parts[0].split(",")]
            output_names = [name.strip() for name in parts[1].split(",")]

            input_fields: dict[str, type] = dict.fromkeys(input_names, str)
            output_fields: dict[str, type] = dict.fromkeys(output_names, str)

            self.signature = make_signature(input_fields, output_fields, "")
        else:
            self.signature = signature

        self.max_iters = max_iters
        self.max_failures = max_failures
        self.enable_ask_to_user = enable_ask_to_user

        # Convert tools to Tool objects
        tool_list = [t if isinstance(t, Tool) else Tool(t) for t in tools]
        self.tools: dict[str, Tool] = {tool.name: tool for tool in tool_list}

        # Track which tools need confirmation
        self.tools_needing_confirmation = {
            name
            for name, tool in self.tools.items()
            if getattr(tool, "ask_for_confirmation", False)
        }

        # Build instruction for the agent
        inputs = ", ".join([f"`{k}`" for k in self.signature.get_input_fields().keys()])
        outputs = ", ".join([f"`{k}`" for k in self.signature.get_output_fields().keys()])

        base_instructions = getattr(self.signature, "__doc__", "")
        instr = [f"{base_instructions}\n"] if base_instructions else []

        instr.extend(
            [
                f"You are an Agent. You will be given {inputs} as input.",
                f"Your goal is to use the supplied tools to accomplish the task and produce {outputs}.\n",
                "Think step-by-step about what to do next, then call the appropriate tool.",
                "Always explain your reasoning before calling a tool.",
                "When you have enough information, call the 'finish' tool to complete the task.",
            ]
        )

        # Add finish tool
        def finish_tool(**_kwargs: Any) -> str:  # pyright: ignore[reportUnusedParameter]
            """Finish tool that accepts and ignores any arguments."""
            return "Task completed"

        self.tools["finish"] = Tool(
            func=finish_tool,
            name="finish",
            desc=f"Call this when you have all information needed to produce {outputs}",
            args={},
        )

        # Add ask_to_user tool if enabled
        if self.enable_ask_to_user:

            def ask_to_user_impl(question: str) -> str:
                """Ask the user for clarification."""
                raise HumanInTheLoopRequired(
                    question=question,
                    tool_name="ask_to_user",
                    tool_args={"question": question},
                )

            self.tools["ask_to_user"] = Tool(
                func=ask_to_user_impl,
                name="ask_to_user",
                description=(
                    "Ask the user for clarification. ONLY use this when:\n"
                    "1. At the very beginning if the request is ambiguous or missing critical information\n"
                    "2. After multiple tool failures when you need help to proceed\n"
                    "Can only be used ONCE per task."
                ),
            )

        # Create ReAct signature with trajectory and reasoning
        # Build input fields: original inputs + trajectory
        react_input_fields: dict[str, type] = {}
        for name, field_info in self.signature.get_input_fields().items():
            # Get the type from the field annotation
            react_input_fields[name] = field_info.annotation or str

        react_input_fields["trajectory"] = str

        # Build output fields: just reasoning (tools will be called natively)
        react_output_fields: dict[str, type] = {
            "reasoning": str,
        }

        self.react_signature = make_signature(
            react_input_fields,
            react_output_fields,
            "\n".join(instr),
        )

        # Create extraction signature
        extract_input_fields: dict[str, type] = {}
        extract_output_fields: dict[str, type] = {}

        # Add original signature fields
        for name, field_info in self.signature.get_input_fields().items():
            extract_input_fields[name] = field_info.annotation or str

        for name, field_info in self.signature.get_output_fields().items():
            extract_output_fields[name] = field_info.annotation or str

        # Add trajectory as input
        extract_input_fields["trajectory"] = str

        self.extract_signature = make_signature(
            extract_input_fields,
            extract_output_fields,
            base_instructions or "Extract the final answer from the trajectory",
        )

        # Create Predict module with native tool calling
        self.react_module = Predict(self.react_signature, tools=list(self.tools.values()))
        self.extract_module = ChainOfThought(self.extract_signature)

    def _format_trajectory(self, trajectory: dict[str, Any]) -> str:
        """Format trajectory as a string for the LLM.

        Args:
            trajectory: Dictionary with reasoning_N, tool_name_N, tool_args_N, observation_N keys

        Returns:
            Formatted string representation
        """
        if not trajectory:
            return "No actions taken yet."

        lines = []
        iteration = 0
        while f"observation_{iteration}" in trajectory:  # Check observation instead
            lines.append(f"\n--- Step {iteration + 1} ---")
            if f"reasoning_{iteration}" in trajectory:
                lines.append(f"Reasoning: {trajectory[f'reasoning_{iteration}']}")
            if f"tool_name_{iteration}" in trajectory:
                lines.append(f"Tool: {trajectory[f'tool_name_{iteration}']}")
            if f"tool_args_{iteration}" in trajectory:
                lines.append(f"Args: {json.dumps(trajectory[f'tool_args_{iteration}'])}")
            lines.append(f"Observation: {trajectory[f'observation_{iteration}']}")
            iteration += 1

        return "\n".join(lines)

    async def _execute_iteration(
        self,
        idx: int,
        trajectory: dict[str, Any],
        input_args: dict[str, Any],
        consecutive_failures: int,
        ask_to_user_used: bool,
        *,
        pending_tool_call: dict[str, Any] | None = None,
    ) -> tuple[bool, int, bool]:
        """Execute a single ReAct iteration.

        Args:
            idx: Current iteration index
            trajectory: Current trajectory state
            input_args: Original input arguments
            consecutive_failures: Count of consecutive tool failures
            ask_to_user_used: Whether ask_to_user has been used
            pending_tool_call: Optional pending tool call to execute (for resumption)
                             Format: {"name": str, "args": dict, "id": str}

        Returns:
            Tuple of (should_stop, new_consecutive_failures, new_ask_to_user_used)

        Raises:
            HumanInTheLoopRequired: When human input is needed
        """
        # If we have a pending tool call (from confirmation/resumption), execute it directly
        if pending_tool_call:
            tool_name = pending_tool_call["name"]
            tool_args = pending_tool_call["args"]
            tool_call_id = pending_tool_call.get("id", "")

            trajectory[f"tool_name_{idx}"] = tool_name
            trajectory[f"tool_args_{idx}"] = tool_args

            # Execute the confirmed/modified tool
            # Bypass confirmation check since this is already confirmed
            try:
                tool = self.tools[tool_name]
                # Call the underlying function directly to bypass confirmation check
                if inspect.iscoroutinefunction(tool.func):
                    observation = await tool.func(**tool_args)
                else:
                    import asyncio

                    loop = asyncio.get_event_loop()
                    observation = await loop.run_in_executor(None, lambda: tool.func(**tool_args))
                consecutive_failures = 0
            except HumanInTheLoopRequired as e:
                # Enrich and re-raise
                e.trajectory = trajectory.copy()
                e.iteration = idx
                e.input_args = input_args.copy()
                e.tool_call_id = tool_call_id
                raise
            except Exception as e:
                observation = f"Error executing {tool_name}: {str(e)}"
                consecutive_failures += 1
                logger.warning(f"Tool execution failed: {e}")

            trajectory[f"observation_{idx}"] = str(observation)
            should_stop = tool_name == "finish"
            return should_stop, consecutive_failures, ask_to_user_used

        # Normal iteration: get LLM decision and execute
        formatted_trajectory = self._format_trajectory(trajectory)
        pred = await self.react_module.aforward(
            **input_args,
            trajectory=formatted_trajectory,
            auto_execute_tools=False,
        )

        # Extract reasoning
        reasoning = pred.get("reasoning", "")
        trajectory[f"reasoning_{idx}"] = reasoning

        # Check if tools were called
        if "tool_calls" not in pred or not pred.tool_calls:
            logger.debug(
                f"No tool calls in prediction. Keys: {list(pred.keys())}, pred: {dict(pred)}"
            )
            raise ValueError("LLM did not call any tools")

        # Get the first tool call (for simplicity, handle one at a time)
        tool_call = pred.tool_calls[0]
        tool_name = tool_call.get("name", "")
        tool_call_id = tool_call.get("id", "")

        # Parse arguments (they come as a JSON string)
        tool_args_str = tool_call.get("arguments", "{}")
        try:
            tool_args = json.loads(tool_args_str) if tool_args_str else {}
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool arguments: {tool_args_str}")
            tool_args = {}

        logger.debug(f"Tool call - name: {tool_name}, args: {tool_args}, id: {tool_call_id}")

        trajectory[f"tool_name_{idx}"] = tool_name
        trajectory[f"tool_args_{idx}"] = tool_args

        # Execute the tool
        try:
            if tool_name == "ask_to_user":
                # Check if ask_to_user already used
                if ask_to_user_used:
                    observation = "Error: ask_to_user can only be used once per task"
                # Check if it's allowed (beginning or after failures)
                elif idx > 0 and consecutive_failures < self.max_failures:
                    observation = f"Error: ask_to_user can only be used at the beginning or after {self.max_failures} consecutive tool failures"
                else:
                    ask_to_user_used = True
                    # Tool will raise HumanInTheLoopRequired - we catch and enrich it
                    tool = self.tools[tool_name]
                    observation = await tool.acall(**tool_args)
            else:
                # Normal tool execution
                tool = self.tools[tool_name]
                observation = await tool.acall(**tool_args)
                consecutive_failures = 0  # Reset on success

        except HumanInTheLoopRequired as e:
            # Enrich exception with trajectory and state
            e.trajectory = trajectory.copy()
            e.iteration = idx
            e.input_args = input_args.copy()
            e.tool_call_id = tool_call_id
            # Re-raise to caller
            raise
        except Exception as e:
            observation = f"Error executing {tool_name}: {str(e)}"
            consecutive_failures += 1
            logger.warning(f"Tool execution failed: {e}")

        trajectory[f"observation_{idx}"] = str(observation)

        # Check if done
        should_stop = tool_name == "finish"
        return should_stop, consecutive_failures, ask_to_user_used

    async def aforward(
        self,
        **input_args: Any,
    ) -> Prediction:
        """Async forward pass with ReAct loop using native tool calling.

        Args:
            **input_args: Input values matching signature's input fields

        Returns:
            Prediction with trajectory and output fields

        Raises:
            HumanInTheLoopRequired: When human input is needed
        """
        max_iters = input_args.pop("max_iters", self.max_iters)
        trajectory: dict[str, Any] = {}
        consecutive_failures = 0
        ask_to_user_used = False

        for idx in range(max_iters):
            try:
                should_stop, consecutive_failures, ask_to_user_used = await self._execute_iteration(
                    idx,
                    trajectory,
                    input_args,
                    consecutive_failures,
                    ask_to_user_used,
                )
                if should_stop:
                    break

            except ValueError as e:
                # Invalid tool selection
                logger.warning(f"Agent failed to select valid tool: {e}")
                trajectory[f"observation_{idx}"] = f"Error: {e}"
                break

        # Extract final answer from trajectory
        formatted_trajectory = self._format_trajectory(trajectory)
        extract = await self.extract_module.aforward(
            **input_args,
            trajectory=formatted_trajectory,
        )

        # Build result with trajectory
        result_dict = {"trajectory": trajectory}
        for field_name in self.signature.get_output_fields():
            if hasattr(extract, field_name):
                result_dict[field_name] = getattr(extract, field_name)

        return Prediction(**result_dict)

    def resume_after_user_input(
        self,
        user_response: str,
        saved_state: HumanInTheLoopRequired,
    ) -> Prediction:
        """Resume execution after user provides input.

        Args:
            user_response: The user's response to the question
            saved_state: The HumanInTheLoopRequired exception that was raised

        Returns:
            Final prediction after resuming execution
        """
        import asyncio

        return asyncio.run(self.aresume_after_user_input(user_response, saved_state))

    async def aresume_after_user_input(
        self,
        user_response: str,
        saved_state: HumanInTheLoopRequired,
    ) -> Prediction:
        """Async resume execution after user input.

        Args:
            user_response: The user's response. Can be:
                - "yes"/"y" to confirm tool execution with original args
                - "no"/"n" to reject and continue
                - JSON dict string to execute tool with modified args
                - Any other text is treated as user feedback for LLM to re-reason

        Returns:
            Final prediction after resuming

        Raises:
            HumanInTheLoopRequired: If another human input is needed
        """
        # Restore state
        trajectory = saved_state.trajectory.copy()
        start_idx = saved_state.iteration or 0
        input_args = saved_state.input_args.copy()
        consecutive_failures = 0
        ask_to_user_used = saved_state.tool_name == "ask_to_user"

        # Determine what to do based on user response
        user_response_lower = user_response.lower().strip()
        pending_tool_call: dict[str, Any] | None = None

        if user_response_lower in ("yes", "y"):
            # User confirmed - execute tool with original args
            pending_tool_call = {
                "name": saved_state.tool_name or "",
                "args": saved_state.tool_args.copy(),
                "id": saved_state.tool_call_id or "",
            }
        elif user_response_lower in ("no", "n"):
            # User rejected - add rejection to trajectory and continue
            trajectory[f"observation_{start_idx}"] = "User rejected the operation"
            start_idx += 1  # Move to next iteration
        else:
            # Try to parse as JSON (modified args)
            try:
                modified_args = json.loads(user_response)
                if isinstance(modified_args, dict):
                    # User provided modified args - execute with those
                    pending_tool_call = {
                        "name": saved_state.tool_name or "",
                        "args": modified_args,
                        "id": saved_state.tool_call_id or "",
                    }
                else:
                    # Not a dict, treat as feedback
                    trajectory[f"observation_{start_idx}"] = f"User feedback: {user_response}"
                    start_idx += 1
            except json.JSONDecodeError:
                # Not JSON, treat as user feedback for LLM to consider
                trajectory[f"observation_{start_idx}"] = f"User feedback: {user_response}"
                start_idx += 1

        # Continue execution from start_idx
        for idx in range(start_idx, self.max_iters):
            try:
                should_stop, consecutive_failures, ask_to_user_used = await self._execute_iteration(
                    idx,
                    trajectory,
                    input_args,
                    consecutive_failures,
                    ask_to_user_used,
                    pending_tool_call=pending_tool_call,
                )
                # Clear pending tool call after first iteration
                pending_tool_call = None

                if should_stop:
                    break

            except ValueError as e:
                logger.warning(f"Agent failed: {e}")
                trajectory[f"observation_{idx}"] = f"Error: {e}"
                break

        # Extract final answer
        formatted_trajectory = self._format_trajectory(trajectory)
        extract = await self.extract_module.aforward(
            **input_args,
            trajectory=formatted_trajectory,
        )

        result_dict = {"trajectory": trajectory}
        for field_name in self.signature.get_output_fields():
            if hasattr(extract, field_name):
                result_dict[field_name] = getattr(extract, field_name)

        return Prediction(**result_dict)
