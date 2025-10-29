"""Message building functions for Predict module."""

from typing import Any

from udspy.adapter import ChatAdapter
from udspy.history import History
from udspy.signature import Signature
from udspy.streaming import Prediction


def build_initial_messages(
    adapter: ChatAdapter,
    signature: type[Signature],
    inputs: dict[str, Any],
    history: History,
) -> None:
    """Build initial system and user messages from inputs and history.

    Adds system message with formatted instructions (if not already present)
    and user message with formatted inputs to the history.

    Args:
        adapter: ChatAdapter for formatting messages
        signature: Signature defining the task
        inputs: Input values from user
        history: History object to update with messages
    """
    if not history.messages:
        history.add_system_message(adapter.format_instructions(signature))

    history.add_user_message(adapter.format_inputs(signature, inputs))


def update_history_with_prediction(
    signature: type[Signature],
    history: History,
    prediction: Prediction,
) -> None:
    """Update history with assistant's final prediction.

    Formats the prediction outputs with field markers and adds as
    assistant message to history.

    Args:
        signature: Signature defining output fields
        history: History object to update
        prediction: Prediction from assistant
    """
    output_fields = signature.get_output_fields()
    content_parts = []

    for field_name in output_fields:
        if hasattr(prediction, field_name):
            value = getattr(prediction, field_name)
            if value:
                content_parts.append(f"[[ ## {field_name} ## ]]\n{value}")

    content = "\n".join(content_parts) if content_parts else ""
    history.add_assistant_message(content)


__all__ = ["build_initial_messages", "update_history_with_prediction"]
