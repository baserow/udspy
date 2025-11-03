"""Tests for @tool decorator and automatic tool execution."""

import pytest
from conftest import make_mock_response
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as CompletionChoice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import Field

from udspy import InputField, OutputField, Predict, Signature, settings, tool


@tool(name="Calculator", description="Perform arithmetic operations")
def calculator(
    operation: str = Field(description="add, subtract, multiply, divide"),
    a: float = Field(description="First number"),
    b: float = Field(description="Second number"),
) -> float:
    """Test calculator tool."""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else float("inf"),
    }
    return ops[operation]


class QA(Signature):
    """Answer questions."""

    question: str = InputField()
    answer: str = OutputField()


def test_tool_decorator() -> None:
    """Test @tool decorator creates Tool object."""
    assert calculator.name == "Calculator"
    assert calculator.description == "Perform arithmetic operations"
    assert "operation" in calculator.args_schema["properties"]
    assert "a" in calculator.args_schema["properties"]
    assert "b" in calculator.args_schema["properties"]


def test_tool_schema_properties() -> None:
    """Test Tool schema properties (input_schema, parameters)."""
    # Test input_schema returns resolved schema
    input_schema = calculator.input_schema
    assert input_schema["type"] == "object"
    assert "properties" in input_schema
    assert "operation" in input_schema["properties"]
    assert "a" in input_schema["properties"]
    assert "b" in input_schema["properties"]
    assert "required" in input_schema
    assert set(input_schema["required"]) == {"operation", "a", "b"}

    # Test parameters returns same as input_schema
    parameters = calculator.parameters
    assert parameters == input_schema
    assert parameters["type"] == "object"
    assert "operation" in parameters["properties"]

    # Test deprecated args_schema still works
    args_schema = calculator.args_schema
    assert args_schema == input_schema

    # Test deprecated args property (just properties section)
    args = calculator.args
    assert args == input_schema["properties"]
    assert "operation" in args


def test_tool_format() -> None:
    """Test Tool.format() returns human-readable string."""
    formatted = calculator.format()
    assert isinstance(formatted, str)
    assert "Calculator" in formatted
    assert "Perform arithmetic operations" in formatted
    # Should mention it takes arguments
    assert "arguments" in formatted.lower() or "takes" in formatted.lower()

    # Test __str__ uses format()
    str_repr = str(calculator)
    assert str_repr == formatted


def test_tool_to_openai_schema() -> None:
    """Test Tool converts to OpenAI schema via adapter."""
    from udspy.adapter import ChatAdapter

    adapter = ChatAdapter()
    schema = adapter.format_tool_schema(calculator)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "Calculator"
    assert schema["function"]["description"] == "Perform arithmetic operations"

    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "operation" in params["properties"]
    assert "a" in params["properties"]
    assert "b" in params["properties"]
    assert set(params["required"]) == {"operation", "a", "b"}


def test_tool_callable() -> None:
    """Test Tool is callable."""
    result = calculator(operation="multiply", a=5, b=3)
    assert result == 15


@pytest.mark.asyncio
async def test_predict_with_tool_automatic_execution() -> None:
    """Test Predict automatically executes tools and handles multi-turn."""

    # Mock first response - LLM requests tool
    first_response = ChatCompletion(
        id="test1",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(
                                name="Calculator",
                                arguments='{"operation": "multiply", "a": 5, "b": 3}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    # Mock second response - LLM provides final answer after seeing tool result
    second_response = make_mock_response('{"answer": "The answer is 15"}')

    call_count = 0

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call - return tool call
            return first_response
        else:
            # Second call - return final answer
            return second_response

    # Mock the client
    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    # Create predictor with tool
    predictor = Predict(QA, tools=[calculator])

    # Call predictor - should automatically handle tool execution
    result = await predictor.aforward(question="What is 5 times 3?")

    # Verify the result
    assert result.answer == "The answer is 15"

    # Verify two API calls were made
    assert call_count == 2


def test_predict_stores_tool_schemas() -> None:
    """Test Predict stores tool schemas correctly."""
    predictor = Predict(QA, tools=[calculator])

    # Verify tool is stored in tools dict
    assert "Calculator" in predictor.tools
    assert predictor.tools["Calculator"] == calculator

    # Verify the tool schema is in tool_schemas
    assert len(predictor.tool_schemas) == 1
    tool_schema = predictor.tool_schemas[0]
    assert tool_schema["type"] == "function"
    assert tool_schema["function"]["name"] == "Calculator"


def test_tool_with_optional_types() -> None:
    """Test Tool handles Optional types correctly."""

    from pydantic import Field

    from udspy.adapter import ChatAdapter

    @tool(name="OptionalTool", description="Tool with optional params")
    def optional_tool(
        required: str = Field(description="Required param"),
        optional: str | None = Field(default=None, description="Optional param"),
    ) -> str:
        return f"{required}-{optional}"

    adapter = ChatAdapter()
    schema = adapter.format_tool_schema(optional_tool)

    # Check that types are converted correctly
    params = schema["function"]["parameters"]
    assert "required" in params["properties"]
    assert "optional" in params["properties"]

    # Required should be in required list
    assert "required" in params["required"]


@pytest.mark.asyncio
async def test_tool_error_handling() -> None:
    """Test error handling when tool execution fails."""

    @tool(name="FailingTool", description="A tool that fails")
    def failing_tool(value: int = Field(description="A value")) -> int:
        raise ValueError("Tool failed!")

    # Mock first response - LLM requests failing tool
    first_response = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_456",
                            type="function",
                            function=Function(
                                name="FailingTool",
                                arguments='{"value": 42}',
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    # Mock second response - LLM provides answer after seeing error
    second_response = make_mock_response('{"answer": "The tool encountered an error"}')

    call_count = 0
    messages_log = []

    async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        messages_log.append(kwargs.get("messages", []))

        if call_count == 1:
            return first_response
        else:
            return second_response

    mock_aclient = settings.lm.client
    mock_aclient.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[failing_tool])

    # Should handle the error gracefully
    result = await predictor.aforward(question="Test")

    # Verify it got an answer even though tool failed
    assert "error" in result.answer.lower()

    # Verify second call included error in tool message
    second_call_messages = messages_log[1]
    tool_message = [m for m in second_call_messages if m["role"] == "tool"][0]
    assert "Error executing tool" in tool_message["content"]


def test_parse_and_validate_args_primitives() -> None:
    """Test parse_and_validate_args with primitive type coercion."""

    @tool(name="TestTool")
    def test_tool(
        int_param: int = Field(description="An integer"),
        float_param: float = Field(description="A float"),
        str_param: str = Field(description="A string"),
        bool_param: bool = Field(description="A boolean"),
    ) -> str:
        return f"{int_param}-{float_param}-{str_param}-{bool_param}"

    # Raw args from JSON (strings that need coercion)
    raw_args = {
        "int_param": "123",
        "float_param": "45.67",
        "str_param": "hello",
        "bool_param": "true",
    }

    parsed = test_tool.parse_and_validate_args(raw_args)

    # Verify types are coerced correctly
    assert isinstance(parsed["int_param"], int)
    assert parsed["int_param"] == 123

    assert isinstance(parsed["float_param"], float)
    assert parsed["float_param"] == 45.67

    assert isinstance(parsed["str_param"], str)
    assert parsed["str_param"] == "hello"

    assert isinstance(parsed["bool_param"], bool)
    assert parsed["bool_param"] is True


def test_parse_and_validate_args_pydantic_models() -> None:
    """Test parse_and_validate_args with Pydantic model arguments."""
    from pydantic import BaseModel

    class RowModel(BaseModel):
        name: str
        value: int
        active: bool = True

    @tool(name="create_row")
    def create_row(
        table_id: int = Field(description="Table ID"),
        row: RowModel = Field(description="Row data"),
    ) -> str:
        return f"Created row {row.name} in table {table_id}"

    # Raw args from JSON
    raw_args = {
        "table_id": "888",  # String that should be coerced to int
        "row": {"name": "Project A", "value": 42},  # Dict that should become RowModel
    }

    parsed = create_row.parse_and_validate_args(raw_args)

    # Verify table_id is coerced to int
    assert isinstance(parsed["table_id"], int)
    assert parsed["table_id"] == 888

    # Verify row is parsed into RowModel instance (not dict!)
    assert isinstance(parsed["row"], RowModel)
    assert parsed["row"].name == "Project A"
    assert parsed["row"].value == 42
    assert parsed["row"].active is True  # Default value


def test_parse_and_validate_args_validation_errors() -> None:
    """Test parse_and_validate_args raises ValidationError for invalid args."""
    from pydantic import ValidationError

    @tool(name="strict_tool")
    def strict_tool(
        required_int: int = Field(description="Must be an int"),
    ) -> str:
        return f"{required_int}"

    # Invalid arg that can't be coerced to int
    raw_args = {"required_int": "not_a_number"}

    with pytest.raises(ValidationError):
        strict_tool.parse_and_validate_args(raw_args)


def test_parse_and_validate_args_nested_pydantic() -> None:
    """Test parse_and_validate_args with nested Pydantic models."""
    from pydantic import BaseModel

    class Address(BaseModel):
        street: str
        city: str
        zip_code: str

    class User(BaseModel):
        name: str
        age: int
        address: Address

    @tool(name="create_user")
    def create_user(
        user: User = Field(description="User data"),
    ) -> str:
        return f"Created user {user.name}"

    # Raw args with nested dict structure
    raw_args = {
        "user": {
            "name": "John Doe",
            "age": "30",  # String that should be coerced
            "address": {"street": "123 Main St", "city": "NYC", "zip_code": "10001"},
        }
    }

    parsed = create_user.parse_and_validate_args(raw_args)

    # Verify user is parsed into User instance
    assert isinstance(parsed["user"], User)
    assert parsed["user"].name == "John Doe"
    assert parsed["user"].age == 30  # Coerced to int

    # Verify nested address is also a model instance
    assert isinstance(parsed["user"].address, Address)
    assert parsed["user"].address.city == "NYC"
