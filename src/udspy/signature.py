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

        # Validate that all fields are marked as input or output
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
