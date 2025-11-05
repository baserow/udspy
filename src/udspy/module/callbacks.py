"""Module callback infrastructure for dynamic tool management."""

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from udspy import History
    from udspy.module.base import Module
    from udspy.module.predict import Predict
    from udspy.module.react import Episode, ReAct


class ModuleCallback:
    """Wrapper for module callback functions.

    Module callbacks are special callables that can be returned by tools
    to dynamically modify the module during execution. They receive a
    context object with the module instance and execution state.

    Example:
        ```python
        from udspy import tool, module_callback

        @tool(name="LoadTools", description="Load additional tools")
        def load_tools(category: str):
            @module_callback
            def callback(context):
                # Add new tools based on category
                new_tools = get_tools_for_category(category)
                context.module.init_module(
                    tools=list(context.module.tools.values()) + new_tools
                )
                return f"Loaded {len(new_tools)} tools for {category}"

            return callback
        ```
    """

    def __init__(self, func: Callable[..., str]):
        """Initialize module callback.

        Args:
            func: Callable that takes a context and returns an observation string
        """
        self.func = func
        wraps(func)(self)

    def __call__(self, context: "ModuleContext") -> str:
        """Execute the callback.

        Args:
            context: Module context with execution state

        Returns:
            Observation or tool result string
        """
        return self.func(context)


def module_callback(func: Callable[..., str]) -> ModuleCallback:
    """Decorator to mark a callable as a module callback.

    Module callbacks can modify the module during execution by calling
    methods like `init_module()` on the module instance available in
    the context.

    Args:
        func: Function that takes a ModuleContext and returns a string

    Returns:
        ModuleCallback wrapper

    Example:
        ```python
        @module_callback
        def add_specialized_tools(context):
            new_tools = [weather_tool, calendar_tool]
            all_tools = list(context.module.tools.values()) + new_tools
            context.module.init_module(tools=all_tools)
            return "Added weather and calendar tools"
        ```
    """
    return ModuleCallback(func)


def is_module_callback(obj: Any) -> bool:
    """Check if an object is a module callback.

    Args:
        obj: Object to check

    Returns:
        True if obj is a ModuleCallback instance
    """
    return isinstance(obj, ModuleCallback)


class ModuleContext:
    """Base context passed to module callbacks.

    Provides access to the module instance so callbacks can modify
    module state during execution.

    Attributes:
        module: The module instance (Predict or ReAct)
    """

    def __init__(self, module: "Module"):
        """Initialize module context.

        Args:
            module: The module instance
        """
        self.module = module


class ReactContext(ModuleContext):
    """Context for ReAct module callbacks and confirmations.

    Provides access to the module, trajectory, and input arguments,
    allowing callbacks to inspect the agent's reasoning history and
    providing all necessary state for confirmation resumption.

    Attributes:
        module: The ReAct module instance
        trajectory: List of completed episodes
        input_args: Original input arguments to the module
        stream: Whether streaming is enabled
    """

    def __init__(
        self,
        module: "ReAct",
        trajectory: list["Episode"],
        input_args: dict[str, Any],
        history: Optional["History"] = None,
        stream: bool = False,
    ):
        """Initialize ReAct context.

        Args:
            module: The ReAct module instance
            trajectory: Current trajectory (list of episodes)
            input_args: Original input arguments
            stream: Whether streaming is enabled
        """
        super().__init__(module)
        self.trajectory = trajectory
        self.input_args = input_args
        self.stream = stream
        self.history = history


class PredictContext(ModuleContext):
    """Context for Predict module callbacks.

    Provides access to both the module and the conversation history,
    allowing callbacks to inspect past interactions.

    Attributes:
        module: The Predict module instance
        history: Conversation history (if provided)
    """

    def __init__(self, module: "Predict", history: Optional["History"] = None):
        """Initialize Predict context.

        Args:
            module: The Predict module instance
            history: Conversation history (if any)
        """
        super().__init__(module)
        self.history = history
