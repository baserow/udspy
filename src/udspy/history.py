"""Conversation history management for multi-turn interactions."""

from typing import Any


class History:
    """Manages conversation history for multi-turn interactions.

    History stores messages in OpenAI format and provides methods to add
    user messages, assistant responses, and tool interactions. When passed
    to Predict, it automatically handles message history.

    Example:
        ```python
        from udspy import History, Predict, Signature, InputField, OutputField

        class QA(Signature):
            '''Answer questions.'''
            question: str = InputField()
            answer: str = OutputField()

        predictor = Predict(QA)
        history = History()

        # First turn
        result = predictor(question="What is Python?", history=history)
        print(result.answer)

        # Second turn - history is automatically maintained
        result = predictor(question="What are its main features?", history=history)
        print(result.answer)  # Uses context from previous turn

        # Access messages
        print(history.messages)  # List of all messages
        ```

    Attributes:
        messages: List of conversation messages in OpenAI format
    """

    def __init__(self, messages: list[dict[str, Any]] | None = None):
        """Initialize conversation history.

        Args:
            messages: Optional initial messages in OpenAI format
        """
        self.messages: list[dict[str, Any]] = messages or []

    def add_message(
        self, role: str, content: str, *, tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        """Add a message to the history.

        Args:
            role: Message role ("system", "user", "assistant", "tool")
            content: Message content
            tool_calls: Optional tool calls for assistant messages
        """
        message: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def add_user_message(self, content: str) -> None:
        """Add a user message.

        Args:
            content: User message content
        """
        self.add_message("user", content)

    def add_assistant_message(
        self, content: str = "", tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        """Add an assistant message.

        Args:
            content: Assistant message content
            tool_calls: Optional tool calls
        """
        self.add_message("assistant", content, tool_calls=tool_calls)

    def add_system_message(self, content: str) -> None:
        """Add a system message.

        Args:
            content: System message content
        """
        self.add_message("system", content)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        """Add a tool result message.

        Args:
            tool_call_id: ID of the tool call this result is for
            content: Tool result content
        """
        self.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": content})

    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages.clear()

    def copy(self) -> "History":
        """Create a copy of this history.

        Returns:
            New History instance with copied messages
        """
        return History(messages=[msg.copy() for msg in self.messages])

    def __len__(self) -> int:
        """Get number of messages in history.

        Returns:
            Number of messages
        """
        return len(self.messages)

    def __repr__(self) -> str:
        """String representation of history.

        Returns:
            String showing number of messages
        """
        return f"History({len(self.messages)} messages)"

    def __str__(self) -> str:
        """Human-readable string representation.

        Returns:
            Formatted conversation history
        """
        lines = [f"History ({len(self.messages)} messages):"]
        for i, msg in enumerate(self.messages, 1):
            role = msg["role"]
            content = msg.get("content", "")
            if len(content) > 50:
                content = content[:47] + "..."
            lines.append(f"  {i}. [{role}] {content}")
        return "\n".join(lines)
