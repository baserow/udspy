"""Streaming support with event queue for incremental LLM outputs and tool updates."""

import asyncio
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from udspy import Module
    from udspy.tool import ToolCall

# Context variable for the current stream's event queue
_stream_queue: ContextVar[asyncio.Queue[Any] | None] = ContextVar("_stream_queue", default=None)


class StreamEvent:
    """Base class for all stream events.

    Users can define custom event types by inheriting from this class.
    The only built-in events are OutputStreamChunk and Prediction.

    Example:
        ```python
        from dataclasses import dataclass
        from udspy.streaming import StreamEvent, emit_event

        @dataclass
        class ToolProgress(StreamEvent):
            tool_name: str
            message: str
            progress: float  # 0.0 to 1.0

        # In your tool:
        async def my_tool():
            await emit_event(ToolProgress("search", "Searching...", 0.5))
        ```
    """

    pass


class StreamChunk(StreamEvent):
    """A chunk of streamed output from a Module."""

    module: "Module"
    field_name: str
    delta: str
    content: str
    is_complete: bool

    def __init__(
        self,
        module: "Module",
        field_name: str,
        delta: str,
        content: str,
        is_complete: bool,
    ) -> None:
        self.module = module
        self.field_name = field_name
        self.delta = delta
        self.content = content
        self.is_complete = is_complete

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else "streaming"
        return (
            f"{self.__class__.__name__}(field={self.field_name}, "
            f"status={status}, delta={self.delta!r}, content={self.content!r})"
        )


class OutputStreamChunk(StreamChunk):
    """A chunk of streamed LLM output for a specific field.

    Attributes:
        field_name: Name of the output field
        delta: Incremental content for this field (new text since last chunk)
        content: Full accumulated content for this field so far
        is_complete: Whether this field is finished streaming
    """

    pass


class ThoughtStreamChunk(StreamChunk):
    """A chunk of streamed reasoning output for a specific step.

    Attributes:
        module: The module emitting this chunk

        delta: Incremental content for this step (new text since last chunk)
        content: Full accumulated content for this step so far
        is_complete: Whether this step is finished streaming
    """

    pass


class Prediction(StreamEvent, dict[str, Any]):
    """Final prediction result with attribute access.

    This is both a StreamEvent (can be yielded from astream) and a dict
    (for convenient attribute access to outputs).

    Attributes:
        module: The module that produced this prediction
        is_final: Whether this is the final prediction (vs intermediate from nested module)
        native_tool_calls: Tool calls from native LLM response (if any)

    Example:
        ```python
        pred = Prediction(answer="Paris", reasoning="France's capital")
        print(pred.answer)  # "Paris"
        print(pred["answer"])  # "Paris"
        print(pred.is_final)  # True for top-level result
        print(pred.module)  # Module instance that produced this
        ```
    """

    def __init__(
        self,
        /,
        module: "Module | None" = None,
        is_final: bool = True,
        native_tool_calls: list["ToolCall"] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.module = module
        self.is_final = is_final
        self.native_tool_calls = native_tool_calls

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Prediction has no attribute '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


async def emit_event(event: StreamEvent) -> None:
    """Emit an event to the active stream.

    This can be called from anywhere (tools, callbacks, etc.) to inject
    events into the current streaming context. If no stream is active,
    this is a no-op (silently ignored).

    Args:
        event: The event to emit (any subclass of StreamEvent)

    Example:
        ```python
        from udspy.streaming import emit_event, StreamEvent
        from dataclasses import dataclass

        @dataclass
        class ToolStatus(StreamEvent):
            message: str

        async def my_tool():
            await emit_event(ToolStatus("Starting search..."))
            result = await do_search()
            await emit_event(ToolStatus("Search complete"))
            return result

        # In the stream consumer:
        async for event in predictor.astream(question="..."):
            if isinstance(event, ToolStatus):
                print(f"📊 {event.message}")
            elif isinstance(event, OutputStreamChunk):
                print(event.delta, end="", flush=True)
        ```
    """
    queue = _stream_queue.get()
    if queue is not None:
        await queue.put(event)


__all__ = [
    "StreamEvent",
    "OutputStreamChunk",
    "ThoughtStreamChunk",
    "Prediction",
    "emit_event",
    "_stream_queue",
]
