"""udspy: A minimal DSPy-inspired library with native OpenAI tool calling."""

from udspy.adapter import ChatAdapter
from udspy.module import ChainOfThought, Module, Predict, Prediction
from udspy.settings import settings
from udspy.signature import InputField, OutputField, Signature, make_signature
from udspy.streaming import StreamChunk, StreamingPredict, streamify

__version__ = "0.1.0"

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
    # Adapter
    "ChatAdapter",
    # Streaming
    "StreamingPredict",
    "StreamChunk",
    "streamify",
]
