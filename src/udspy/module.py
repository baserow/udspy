"""Module abstraction for composable LLM calls."""

from typing import Any

from openai.types.chat import ChatCompletion

from udspy.adapter import ChatAdapter
from udspy.settings import settings
from udspy.signature import Signature


class Prediction(dict[str, Any]):
    """Container for prediction outputs with attribute access.

    Example:
        ```python
        pred = Prediction(answer="Paris", reasoning="France's capital")
        print(pred.answer)  # "Paris"
        print(pred["answer"])  # "Paris"
        ```
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"Prediction has no attribute '{name}'") from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class Module:
    """Base class for all udspy modules.

    Modules are composable units that can be called to produce predictions.
    Subclasses should implement the `forward` method.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Prediction:
        """Call the module's forward method.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Prediction object with outputs
        """
        return self.forward(*args, **kwargs)

    def forward(self, *_args: Any, **_kwargs: Any) -> Prediction:
        """Forward pass logic. Must be implemented by subclasses.

        Args:
            *_args: Positional arguments
            **_kwargs: Keyword arguments

        Returns:
            Prediction object with outputs
        """
        raise NotImplementedError


class Predict(Module):
    """Module for making LLM predictions based on a signature.

    Example:
        ```python
        class QA(Signature):
            '''Answer questions.'''
            question: str = InputField()
            answer: str = OutputField()

        predictor = Predict(QA)
        result = predictor(question="What is 2+2?")
        print(result.answer)
        ```
    """

    def __init__(
        self,
        signature: type[Signature],
        *,
        model: str | None = None,
        tools: list[type] | None = None,
        adapter: ChatAdapter | None = None,
        **kwargs: Any,
    ):
        """Initialize a Predict module.

        Args:
            signature: Signature defining inputs and outputs
            model: Model name (overrides global default)
            tools: List of Pydantic tool models for native function calling
            adapter: Custom adapter (defaults to ChatAdapter)
            **kwargs: Additional arguments for chat completion (temperature, etc.)
        """
        self.signature = signature
        self.model = model or settings.default_model
        self.tools = tools or []
        self.adapter = adapter or ChatAdapter()
        self.kwargs = {**settings.default_kwargs, **kwargs}

    def forward(self, **inputs: Any) -> Prediction:
        """Execute the prediction.

        Args:
            **inputs: Input values matching the signature's input fields

        Returns:
            Prediction with outputs

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
            **self.kwargs,
        }

        # Add tools if provided
        if self.tools:
            tool_schemas = self.adapter.format_tool_schemas(self.tools)
            completion_kwargs["tools"] = tool_schemas

        # Make API call
        client = settings.client
        response: ChatCompletion = client.chat.completions.create(**completion_kwargs)

        # Parse response
        completion_text = response.choices[0].message.content or ""
        outputs = self.adapter.parse_outputs(self.signature, completion_text)

        # Handle tool calls if present
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            outputs["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,  # type: ignore[union-attr]
                    "arguments": tc.function.arguments,  # type: ignore[union-attr]
                }
                for tc in tool_calls
            ]

        return Prediction(**outputs)
