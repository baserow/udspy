"""Conversation history management for multi-turn interactions."""

from typing import Any


class History:
    """Manages conversation history for multi-turn interactions.

    History stores messages in OpenAI format and provides methods to add
    user messages, assistant responses, and tool interactions. When passed
    to Predict, it automatically manages the system prompt, ensuring it's
    always the first message.

    The system prompt is automatically set based on the signature when using
    Predict, so you typically only need to track user/assistant messages in
    your History. This makes it easy to maintain conversation context without
    worrying about system prompt placement.

    Example:
        ```python
        from udspy import History, Predict, Signature, InputField, OutputField

        class QA(Signature):
            '''Answer questions.'''
            question: str = InputField()
            answer: str = OutputField()

        predictor = Predict(QA)
        history = History()

        # First turn - system prompt automatically added at position 0
        result = predictor(question="What is Python?", history=history)
        print(result.answer)

        # Second turn - system prompt is maintained, new user message added
        result = predictor(question="What are its main features?", history=history)
        print(result.answer)  # Uses context from previous turn

        # Access messages - system prompt is at position 0
        print(history.messages[0])  # System prompt
        print(history.messages[1])  # First user message
        ```

    You can also manually manage the history with only user/assistant messages:
        ```python
        # Pre-populate history with previous conversation
        history = History()
        history.add_user_message("What is Python?")
        history.add_assistant_message("Python is a programming language")

        # System prompt will be prepended automatically when passed to Predict
        result = predictor(question="What are its features?", history=history)
        # history.messages[0] is now the system prompt
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

    def set_system_message(self, content: str) -> None:
        """Set or replace the system message at the beginning of history.

        This ensures the system message is always the first message in the
        conversation history. If a system message already exists at position 0,
        it will be replaced. Otherwise, the system message will be prepended.

        This is particularly useful when working with the Predict module, which
        automatically sets the system prompt based on the signature. Users can
        maintain a history with only user/assistant messages, and the system
        prompt will be automatically managed.

        Args:
            content: System message content

        Example:
            ```python
            # History with user messages only
            history = History()
            history.add_user_message("What is Python?")
            history.add_assistant_message("Python is a programming language")

            # System message is prepended when passed to Predict
            # (handled automatically by the module)
            result = predictor(question="What are its features?", history=history)
            # history.messages[0] is now the system message
            ```
        """
        message = {"role": "system", "content": content}

        if not self.messages:
            self.messages.append(message)
        elif self.messages[0]["role"] == "system":
            self.messages[0] = message
        else:
            self.messages.insert(0, message)

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
