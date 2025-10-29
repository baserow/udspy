"""Language model abstraction layer."""

from udspy.lm.base import LM
from udspy.lm.openai import OpenAILM

__all__ = ["LM", "OpenAILM"]
