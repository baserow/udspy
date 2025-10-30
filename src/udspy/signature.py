"""Signature definitions for structured LLM inputs and outputs."""

from typing import Any

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo


def InputField(
    default: Any = ...,
    *,
    description: str | None = None,
    **kwargs: Any,
) -> Any:
    """Define an input field for a Signature.

    Args:
        default: Default value for the field
        description: Human-readable description of the field's purpose
        **kwargs: Additional Pydantic field arguments

    Returns:
        A Pydantic FieldInfo with input metadata
    """
    json_schema_extra = kwargs.pop("json_schema_extra", {})
    json_schema_extra["__udspy_field_type"] = "input"

    return Field(
        default=default,
        description=description,
        json_schema_extra=json_schema_extra,
        **kwargs,
    )


def OutputField(
    default: Any = ...,
    *,
    description: str | None = None,
    **kwargs: Any,
) -> Any:
    """Define an output field for a Signature.

    Args:
        default: Default value for the field
        description: Human-readable description of the field's purpose
        **kwargs: Additional Pydantic field arguments

    Returns:
        A Pydantic FieldInfo with output metadata
    """
    json_schema_extra = kwargs.pop("json_schema_extra", {})
    json_schema_extra["__udspy_field_type"] = "output"

    return Field(
        default=default,
        description=description,
        json_schema_extra=json_schema_extra,
        **kwargs,
    )


class SignatureMeta(type(BaseModel)):  # type: ignore[misc]
    """Metaclass for Signature that validates field types."""

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> type:
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip validation for the base Signature class
        if name == "Signature":
            return cls

        for field_name, field_info in cls.model_fields.items():
            if not isinstance(field_info, FieldInfo):
                continue

            json_schema_extra = field_info.json_schema_extra or {}
            field_type = json_schema_extra.get("__udspy_field_type")  # type: ignore[union-attr]

            if field_type not in ("input", "output"):
                raise TypeError(
                    f"Field '{field_name}' in {name} must be declared with "
                    f"InputField() or OutputField()"
                )

        return cls


class Signature(BaseModel, metaclass=SignatureMeta):
    """Base class for defining LLM task signatures.

    A Signature specifies the input and output fields for an LLM task,
    along with an optional instruction describing the task.

    Example:
        ```python
        class QA(Signature):
            '''Answer questions concisely.'''
            question: str = InputField(description="Question to answer")
            answer: str = OutputField(description="Concise answer")
        ```
    """

    @classmethod
    def get_input_fields(cls) -> dict[str, FieldInfo]:
        """Get all input fields defined in this signature."""
        return {
            name: field_info
            for name, field_info in cls.model_fields.items()
            if (field_info.json_schema_extra or {}).get("__udspy_field_type") == "input"  # type: ignore[union-attr]
        }

    @classmethod
    def get_output_fields(cls) -> dict[str, FieldInfo]:
        """Get all output fields defined in this signature."""
        return {
            name: field_info
            for name, field_info in cls.model_fields.items()
            if (field_info.json_schema_extra or {}).get("__udspy_field_type") == "output"  # type: ignore[union-attr]
        }

    @classmethod
    def get_instructions(cls) -> str:
        """Get the task instructions from the docstring."""
        return (cls.__doc__ or "").strip()

    @classmethod
    def from_string(cls, spec: str, instructions: str = "") -> type["Signature"]:
        """Create a Signature from DSPy-style string format.

        This is a convenience method for creating simple signatures using
        the DSPy string format "input1, input2 -> output1, output2".
        All fields default to type `str`.

        For more control over field types, descriptions, and defaults,
        use the class-based Signature definition or `make_signature()`.

        Args:
            spec: Signature specification in format "inputs -> outputs"
                  Examples: "question -> answer"
                           "context, question -> answer"
                           "text -> summary, keywords"
            instructions: Optional task instructions (docstring)

        Returns:
            A new Signature class with all fields as type `str`

        Raises:
            ValueError: If spec is not in valid format

        Example:
            ```python
            # Simple signature
            QA = Signature.from_string("question -> answer", "Answer questions")
            predictor = Predict(QA)

            # Multiple inputs and outputs
            Summarize = Signature.from_string(
                "document, style -> summary, keywords",
                "Summarize documents in specified style"
            )
            ```

        Note:
            This is equivalent to DSPy's string-based signature creation.
            All fields default to `str` type. For custom types, use the
            class-based approach with InputField() and OutputField().
        """
        if "->" not in spec:
            raise ValueError(
                f"Invalid signature format: '{spec}'. "
                "Must be in format 'inputs -> outputs' (e.g., 'question -> answer')"
            )

        parts = spec.split("->")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid signature format: '{spec}'. Must have exactly one '->' separator"
            )

        input_str = parts[0].strip()
        if not input_str:
            raise ValueError("Signature must have at least one input field")

        input_names = [name.strip() for name in input_str.split(",")]
        input_fields: dict[str, type] = {name: str for name in input_names if name}

        output_str = parts[1].strip()
        if not output_str:
            raise ValueError("Signature must have at least one output field")

        output_names = [name.strip() for name in output_str.split(",")]
        output_fields: dict[str, type] = {name: str for name in output_names if name}

        return make_signature(input_fields, output_fields, instructions)


def make_signature(
    input_fields: dict[str, type],
    output_fields: dict[str, type],
    instructions: str = "",
) -> type[Signature]:
    """Dynamically create a Signature class.

    Args:
        input_fields: Dictionary mapping field names to types for inputs
        output_fields: Dictionary mapping field names to types for outputs
        instructions: Task instructions

    Returns:
        A new Signature class

    Example:
        ```python
        QA = make_signature(
            {"question": str},
            {"answer": str},
            "Answer questions concisely"
        )
        ```
    """
    fields = {}

    for name, type_ in input_fields.items():
        fields[name] = (type_, InputField())

    for name, type_ in output_fields.items():
        fields[name] = (type_, OutputField())

    sig = create_model(
        "DynamicSignature",
        __base__=Signature,
        **fields,  # type: ignore
    )

    if instructions:
        sig.__doc__ = instructions

    return sig
