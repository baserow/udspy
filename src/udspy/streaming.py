"""Streaming support for incremental LLM outputs."""

import re
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from udspy.module import Predict, Prediction
from udspy.settings import settings


class StreamChunk:
    """A chunk of streamed output for a specific field.

    Attributes:
        field_name: Name of the output field
        content: Incremental content for this field
        is_complete: Whether this field is finished streaming
    """

    def __init__(self, field_name: str, content: str, is_complete: bool = False):
        self.field_name = field_name
        self.content = content
        self.is_complete = is_complete

    def __repr__(self) -> str:
        status = "complete" if self.is_complete else "streaming"
        return f"StreamChunk(field={self.field_name}, status={status}, content={self.content!r})"


class StreamingPredict(Predict):
    """Streaming version of Predict that yields incremental outputs.

    Example:
        ```python
        predictor = StreamingPredict(signature)
        async for chunk in predictor.stream(question="What is AI?"):
            print(f"{chunk.field_name}: {chunk.content}", end="", flush=True)
        ```
    """

    async def stream(self, **inputs: Any) -> AsyncIterator[StreamChunk | Prediction]:
        """Stream prediction outputs incrementally.

        Args:
            **inputs: Input values matching the signature's input fields

        Yields:
            StreamChunk objects for each output field, followed by final Prediction

        Raises:
            ValueError: If required inputs are missing
        """
        # Validate inputs
        input_fields = self.signature.get_input_fields()
        for field_name in input_fields:
            if field_name not in inputs:
                raise ValueError(f"Missing required input field: {field_name}")

        # Build messages
        messages = [
            {
                "role": "system",
                "content": self.adapter.format_instructions(self.signature),
            },
            {
                "role": "user",
                "content": self.adapter.format_inputs(self.signature, inputs),
            },
        ]

        # Prepare kwargs
        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.kwargs,
        }

        # Add tools if provided
        if self.tools:
            tool_schemas = self.adapter.format_tool_schemas(self.tools)
            completion_kwargs["tools"] = tool_schemas

        # Make streaming API call
        client = settings.async_client
        stream: AsyncStream[ChatCompletionChunk] = await client.chat.completions.create(
            **completion_kwargs
        )

        # Track streaming state
        output_fields = self.signature.get_output_fields()
        field_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")
        current_field: str | None = None
        accumulated_content: dict[str, list[str]] = {name: [] for name in output_fields}
        full_completion: list[str] = []

        # Process stream
        async for chunk in stream:
            delta = chunk.choices[0].delta
            content = delta.content

            if not content:
                continue

            # Accumulate full completion for final parsing
            full_completion.append(content)

            # Check for field markers
            match = field_pattern.search(content)
            if match:
                # Entering a new field
                if current_field:
                    # Mark previous field as complete
                    yield StreamChunk(current_field, "", is_complete=True)

                current_field = match.group(1)
                # Remove the marker from content
                content = field_pattern.sub("", content)

            # Stream content for current field
            if current_field and current_field in output_fields:
                accumulated_content[current_field].append(content)
                yield StreamChunk(current_field, content, is_complete=False)

        # Mark last field as complete
        if current_field:
            yield StreamChunk(current_field, "", is_complete=True)

        # Parse final outputs
        completion_text = "".join(full_completion)
        outputs = self.adapter.parse_outputs(self.signature, completion_text)

        # Return final prediction
        yield Prediction(**outputs)


def streamify(predictor: Predict) -> StreamingPredict:
    """Convert a Predict module to a StreamingPredict.

    Args:
        predictor: A Predict module to make streaming

    Returns:
        A StreamingPredict with the same configuration

    Example:
        ```python
        predictor = Predict(signature)
        streaming_predictor = streamify(predictor)
        async for chunk in streaming_predictor.stream(question="Hi"):
            print(chunk.content, end="")
        ```
    """
    return StreamingPredict(
        signature=predictor.signature,
        model=predictor.model,
        tools=predictor.tools,
        adapter=predictor.adapter,
        **predictor.kwargs,
    )
