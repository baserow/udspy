"""Chain of Thought reasoning module."""

from typing import Any

from udspy.adapter import ChatAdapter
from udspy.callback import with_callbacks
from udspy.module.base import Module
from udspy.module.predict import Predict
from udspy.signature import Signature, make_signature
from udspy.streaming import Prediction
from udspy.tool import Tool


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
        signature: type[Signature] | str,
        *,
        reasoning_description: str = "Step-by-step reasoning process",
        tools: list[Tool] | None = None,
        model: str | None = None,
        adapter: ChatAdapter | None = None,
        **kwargs: Any,
    ):
        """Initialize a Chain of Thought module.

        Args:
            signature: Signature defining inputs and final outputs, or a string in
                      format "inputs -> outputs" (e.g., "question -> answer")
            reasoning_description: Description for the reasoning field
            model: Model name (overrides global default)
            tools: List of Pydantic tool models
            adapter: Custom adapter
            **kwargs: Additional arguments for chat completion (including callbacks)
        """
        if isinstance(signature, str):
            signature = Signature.from_string(signature)

        self.original_signature = signature
        self.reasoning_description = reasoning_description
        self.model = model
        self.adapter = adapter
        self.kwargs = kwargs

        # Initialize module with tools
        self.init_module(tools=tools)

    def init_module(self, tools: list[Any] | None = None) -> None:
        """Initialize or reinitialize ChainOfThought with new tools.

        Args:
            tools: New tools to initialize with
        """
        extended_signature = self._build_extended_signature()
        self._create_predictor(extended_signature, tools)

    def _build_extended_signature(self) -> type[Signature]:
        """Build extended signature with reasoning field.

        Returns:
            Signature with reasoning field prepended to outputs
        """
        signature = self.original_signature

        input_fields = {
            name: field.annotation for name, field in signature.get_input_fields().items()
        }
        output_fields = {
            name: field.annotation for name, field in signature.get_output_fields().items()
        }

        extended_outputs = {"reasoning": str, **output_fields}

        extended_signature = make_signature(
            input_fields,  # type: ignore[arg-type]
            extended_outputs,  # type: ignore[arg-type]
            signature.get_instructions(),
        )

        extended_signature.model_fields["reasoning"].description = self.reasoning_description

        return extended_signature

    def _create_predictor(self, signature: type[Signature], tools: list[Any] | None) -> None:
        """Create the internal Predict module.

        Args:
            signature: Extended signature with reasoning field
            tools: Tools to pass to Predict
        """
        self.predict = Predict(
            signature, tools=tools, model=self.model, adapter=self.adapter, **self.kwargs
        )

    @with_callbacks
    async def aexecute(self, *, stream: bool = False, **inputs: Any) -> Prediction:
        """Execute chain of thought prediction.

        Delegates to the wrapped Predict module's aexecute method, which will
        automatically emit streaming events if stream=True and a queue is active.

        Args:
            stream: If True, request streaming from LLM provider
            **inputs: Input values matching the signature's input fields

        Returns:
            Prediction with reasoning and other output fields
        """
        return await self.predict.aexecute(stream=stream, **inputs)
