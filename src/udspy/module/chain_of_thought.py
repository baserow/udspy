"""Chain of Thought reasoning module."""

from collections.abc import AsyncGenerator
from typing import Any

from udspy.adapter import ChatAdapter
from udspy.module.base import Module
from udspy.module.predict import Predict
from udspy.signature import Signature, make_signature
from udspy.streaming import StreamEvent


class ChainOfThought(Module):
    """Chain of Thought reasoning module.

    Automatically adds a reasoning step before generating outputs.
    This encourages the LLM to think step-by-step, improving answer quality.

    Example:
        ```python
        class QA(Signature):
            '''Answer questions.'''
            question: str = InputField()
            answer: str = OutputField()

        # Creates predictor with automatic reasoning
        predictor = ChainOfThought(QA)
        result = predictor(question="What is 2+2?")

        print(result.reasoning)  # "Let's think step by step..."
        print(result.answer)     # "4"
        ```
    """

    def __init__(
        self,
        signature: type[Signature],
        *,
        reasoning_description: str = "Step-by-step reasoning process",
        model: str | None = None,
        tools: list[type] | None = None,
        adapter: ChatAdapter | None = None,
        **kwargs: Any,
    ):
        """Initialize a Chain of Thought module.

        Args:
            signature: Signature defining inputs and final outputs
            reasoning_description: Description for the reasoning field
            model: Model name (overrides global default)
            tools: List of Pydantic tool models
            adapter: Custom adapter
            **kwargs: Additional arguments for chat completion
        """
        self.original_signature = signature

        # Create extended signature with reasoning field
        input_fields = {
            name: field.annotation for name, field in signature.get_input_fields().items()
        }
        output_fields = {
            name: field.annotation for name, field in signature.get_output_fields().items()
        }

        # Prepend reasoning to outputs
        extended_outputs = {"reasoning": str, **output_fields}

        # Create new signature with reasoning
        extended_signature = make_signature(
            input_fields,  # type: ignore[arg-type]
            extended_outputs,  # type: ignore[arg-type]
            signature.get_instructions(),
        )

        # Override reasoning field description
        extended_signature.model_fields["reasoning"].description = reasoning_description

        # Create predictor with extended signature
        self.predict = Predict(
            extended_signature,
            model=model,
            tools=tools,
            adapter=adapter,
            **kwargs,
        )

    async def astream(self, **inputs: Any) -> AsyncGenerator[StreamEvent, None]:
        """Stream chain of thought prediction with reasoning.

        Delegates to the wrapped Predict module's astream method, which yields
        StreamEvent objects including reasoning field.

        Args:
            **inputs: Input values matching the signature's input fields

        Yields:
            StreamEvent objects (StreamChunk with reasoning and other fields,
            final Prediction with all outputs including reasoning)
        """
        async for event in self.predict.astream(**inputs):
            yield event
