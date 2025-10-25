"""Module package for composable LLM calls."""

from udspy.confirmation import ConfirmationRequired
from udspy.module.base import Module
from udspy.module.chain_of_thought import ChainOfThought
from udspy.module.predict import Predict
from udspy.module.react import ReAct
from udspy.streaming import Prediction

__all__ = [
    "ChainOfThought",
    "ConfirmationRequired",
    "Module",
    "Predict",
    "Prediction",
    "ReAct",
]
