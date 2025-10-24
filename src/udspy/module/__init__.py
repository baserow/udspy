"""Module package for composable LLM calls."""

from udspy.module.base import Module, Prediction
from udspy.module.chain_of_thought import ChainOfThought
from udspy.module.predict import Predict

__all__ = [
    "Module",
    "Prediction",
    "Predict",
    "ChainOfThought",
]
