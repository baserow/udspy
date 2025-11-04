"""udspy: A minimal DSPy-inspired library with native OpenAI tool calling."""

from importlib.metadata import version

from udspy.adapter import ChatAdapter
from udspy.callback import ACTIVE_CALL_ID, BaseCallback, with_callbacks
from udspy.confirmation import (
    ApprovalData,
    ConfirmationRejected,
    ConfirmationRequired,
    ConfirmationStatus,
    ResumeState,
    confirm_first,
    get_confirmation_status,
    respond_to_confirmation,
)
from udspy.history import History
from udspy.lm import LM, OpenAILM
from udspy.module import (
    ChainOfThought,
    Module,
    Predict,
    Prediction,
    ReAct,
)
from udspy.module.callbacks import (
    ModuleCallback,
    ModuleContext,
    PredictContext,
    ReactContext,
    module_callback,
)
from udspy.settings import settings
from udspy.signature import InputField, OutputField, Signature, make_signature
from udspy.streaming import OutputStreamChunk, StreamEvent, emit_event
from udspy.tool import Tool, ToolCall, ToolCalls, Tools, tool

__version__ = version("udspy")

__all__ = [
    # Settings
    "settings",
    # Language Models
    "LM",
    "OpenAILM",
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
    # Callbacks
    "BaseCallback",
    "with_callbacks",
    "ACTIVE_CALL_ID",
    # Module Callbacks
    "module_callback",
    "ModuleCallback",
    "ModuleContext",
    "PredictContext",
    "ReactContext",
    # Confirmation
    "ConfirmationRequired",
    "ConfirmationRejected",
    "ConfirmationStatus",
    "ApprovalData",
    "ResumeState",
    "confirm_first",
    "get_confirmation_status",
    "respond_to_confirmation",
    # Adapter
    "ChatAdapter",
    # History
    "History",
    # Streaming
    "StreamEvent",
    "OutputStreamChunk",
    "emit_event",
    # Tools
    "Tool",
    "tool",
    "Tools",
    "ToolCall",
    "ToolCalls",
]
