"""Base classes for modules and predictions."""

from typing import Any


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
