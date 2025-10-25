"""udspy: A minimal DSPy-inspired library with native OpenAI tool calling."""

from udspy.adapter import ChatAdapter
from udspy.history import History
from udspy.interrupt import (
    HumanInTheLoopRequired,
    InterruptRejected,
    ToolCall,
    get_interrupt_status,
    set_interrupt_approval,
)
from udspy.module import (
    ChainOfThought,
    Module,
    Predict,
    Prediction,
    ReAct,
)
from udspy.settings import settings
from udspy.signature import InputField, OutputField, Signature, make_signature
from udspy.streaming import StreamChunk, StreamEvent, emit_event
from udspy.tool import Tool, tool
from udspy.utils import asyncify

__version__ = "0.1.1"

__all__ = [
    # Settings
    "settings",
    # Signatures
    "Signature",
    "InputField",
    "OutputField",
    "make_signature",
    # Modules
    "Module",
    "Predict",
    "Prediction",
    "ChainOfThought",
    "ReAct",
    "HumanInTheLoopRequired",
    "InterruptRejected",
    "ToolCall",
    "get_interrupt_status",
    "set_interrupt_approval",
    # Adapter
    "ChatAdapter",
    # History
    "History",
    # Streaming
    "StreamEvent",
    "StreamChunk",
    "emit_event",
    # Tools
    "Tool",
    "tool",
    # Utils
    "asyncify",
]
