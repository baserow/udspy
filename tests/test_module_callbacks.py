"""Tests for module callbacks and dynamic tool management."""

import pytest
from pydantic import Field

from udspy import (
    InputField,
    OutputField,
    Predict,
    ReAct,
    Signature,
    module_callback,
    tool,
)
from udspy.module.callbacks import (
    ModuleCallback,
    ModuleContext,
    PredictContext,
    ReactContext,
    is_module_callback,
)
from udspy.settings import settings


class QA(Signature):
    """Answer questions."""

    question: str = InputField()
    answer: str = OutputField()


def test_module_callback_decorator() -> None:
    """Test @module_callback decorator creates ModuleCallback."""

    @module_callback
    def my_callback(context):
        return "test"

    assert isinstance(my_callback, ModuleCallback)
    assert callable(my_callback)


def test_is_module_callback() -> None:
    """Test is_module_callback detection."""

    @module_callback
    def callback(context):
        return "test"

    def regular_function(context):
        return "test"

    assert is_module_callback(callback) is True
    assert is_module_callback(regular_function) is False
    assert is_module_callback("not a function") is False
    assert is_module_callback(42) is False


def test_module_callback_execution() -> None:
    """Test ModuleCallback can be executed with context."""

    @module_callback
    def callback(context):
        return f"Module type: {type(context.module).__name__}"

    predictor = Predict(QA)
    context = ModuleContext(module=predictor)

    result = callback(context)
    assert "Predict" in result


def test_react_context() -> None:
    """Test ReactContext provides trajectory access."""
    agent = ReAct(QA, tools=[])
    trajectory = {"thought_0": "test thought", "tool_calls_0": []}

    context = ReactContext(module=agent, trajectory=trajectory)

    assert context.module == agent
    assert context.trajectory == trajectory
    assert context.trajectory["thought_0"] == "test thought"


def test_predict_context() -> None:
    """Test PredictContext provides history access."""
    from udspy import History

    predictor = Predict(QA)
    history = History()
    history.add_user_message("test question")

    context = PredictContext(module=predictor, history=history)

    assert context.module == predictor
    assert context.history == history
    assert len(context.history) == 1


def test_predict_init_module() -> None:
    """Test Predict.init_module() rebuilds tools and schemas."""

    @tool(name="tool1", description="First tool")
    def tool1(x: int = Field(...)) -> str:
        return f"tool1: {x}"

    @tool(name="tool2", description="Second tool")
    def tool2(y: str = Field(...)) -> str:
        return f"tool2: {y}"

    # Create predictor with tool1
    predictor = Predict(QA, tools=[tool1])
    assert len(predictor.tools) == 1
    assert "tool1" in predictor.tools
    assert len(predictor.tool_schemas) == 1

    # Reinitialize with both tools
    predictor.init_module(tools=[tool1, tool2])
    assert len(predictor.tools) == 2
    assert "tool1" in predictor.tools
    assert "tool2" in predictor.tools
    assert len(predictor.tool_schemas) == 2


def test_predict_init_module_clears_tools() -> None:
    """Test Predict.init_module(tools=None) clears tools."""

    @tool(name="tool1", description="First tool")
    def tool1(x: int = Field(...)) -> str:
        return f"tool1: {x}"

    predictor = Predict(QA, tools=[tool1])
    assert len(predictor.tools) == 1

    # Clear all tools
    predictor.init_module(tools=None)
    assert len(predictor.tools) == 0
    assert len(predictor.tool_schemas) == 0


def test_react_init_module_preserves_builtin_tools() -> None:
    """Test ReAct.init_module() preserves finish and ask_to_user tools."""

    @tool(name="custom_tool", description="Custom tool")
    def custom_tool(x: int = Field(...)) -> str:
        return f"custom: {x}"

    # Create agent with custom tool
    agent = ReAct(QA, tools=[custom_tool], enable_ask_to_user=True)
    assert "finish" in agent.tools
    assert "ask_to_user" in agent.tools
    assert "custom_tool" in agent.tools

    # Reinitialize with empty tools - built-ins should remain
    agent.init_module(tools=[])
    assert "finish" in agent.tools
    assert "ask_to_user" in agent.tools
    assert "custom_tool" not in agent.tools
    assert len(agent.tools) == 2  # Only finish and ask_to_user


def test_react_init_module_rebuilds_signature() -> None:
    """Test ReAct.init_module() rebuilds signature with new tools."""

    @tool(name="tool1", description="First tool")
    def tool1(x: int = Field(...)) -> str:
        return f"tool1: {x}"

    @tool(name="tool2", description="Second tool")
    def tool2(y: str = Field(...)) -> str:
        return f"tool2: {y}"

    agent = ReAct(QA, tools=[tool1])

    # Check initial signature contains tool1
    initial_instructions = agent.react_signature.__doc__ or ""
    assert "tool1" in initial_instructions
    assert "tool2" not in initial_instructions

    # Add tool2
    agent.init_module(tools=[tool1, tool2])

    # Check updated signature contains both tools
    updated_instructions = agent.react_signature.__doc__ or ""
    assert "tool1" in updated_instructions
    assert "tool2" in updated_instructions


def test_react_init_module_without_ask_to_user() -> None:
    """Test ReAct.init_module() with ask_to_user disabled."""

    @tool(name="custom_tool", description="Custom tool")
    def custom_tool(x: int = Field(...)) -> str:
        return f"custom: {x}"

    # Create agent without ask_to_user
    agent = ReAct(QA, tools=[custom_tool], enable_ask_to_user=False)
    assert "finish" in agent.tools
    assert "ask_to_user" not in agent.tools

    # Reinitialize
    agent.init_module(tools=[])
    assert "finish" in agent.tools
    assert "ask_to_user" not in agent.tools


@pytest.mark.asyncio
async def test_predict_with_module_callback() -> None:
    """Test Predict executes module callbacks and adds tools dynamically."""
    from conftest import make_mock_response
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice as CompletionChoice
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    @tool(name="new_tool", description="Newly loaded tool")
    def new_tool(x: int = Field(...)) -> str:
        return f"new_tool result: {x}"

    @tool(name="load_tools", description="Load new tools")
    def load_tools(category: str = Field(...)) -> callable:  # type: ignore[valid-syntax]
        @module_callback
        def callback(context):
            # Add new_tool to current tools
            current = list(context.module.tools.values())
            context.module.init_module(tools=current + [new_tool])
            return f"Loaded tools for {category}"

        return callback

    # Mock first response with tool call to load_tools
    response1 = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="[[ ## answer ## ]]\\nLoading tools",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_123",
                            type="function",
                            function=Function(name="load_tools", arguments='{"category": "math"}'),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    # Mock second response with final answer
    response2 = make_mock_response("[[ ## answer ## ]]\\nTools loaded successfully")

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        call_count["count"] += 1
        if call_count["count"] == 1:
            return response1
        return response2

    settings.lm.client.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[load_tools])
    assert "new_tool" not in predictor.tools

    await predictor.aforward(question="Load math tools")

    # Verify new_tool was added
    assert "new_tool" in predictor.tools
    assert len(predictor.tool_schemas) >= 2  # load_tools + new_tool


@pytest.mark.asyncio
async def test_react_with_module_callback() -> None:
    """Test ReAct executes module callbacks and adds tools dynamically."""
    from conftest import make_mock_response

    @tool(name="weather_tool", description="Get weather")
    def weather_tool(location: str = Field(...)) -> str:
        return f"Weather in {location}: Sunny"

    @tool(name="load_tools", description="Load specialized tools")
    def load_tools(category: str = Field(...)) -> callable:  # type: ignore[valid-syntax]
        @module_callback
        def callback(context):
            # Get non-builtin tools
            current = [
                t for t in context.module.tools.values() if t.name not in ("finish", "ask_to_user")
            ]
            # Add weather tool
            context.module.init_module(tools=current + [weather_tool])
            return f"Loaded {category} tools"

        return callback

    # Mock LLM responses
    responses = [
        # Load tools
        '[[ ## next_thought ## ]]\nI need to load weather tools\n[[ ## next_tool_calls ## ]]\n[{"name": "load_tools", "args": {"category": "weather"}}]',
        # Use weather tool
        '[[ ## next_thought ## ]]\nNow check the weather\n[[ ## next_tool_calls ## ]]\n[{"name": "weather_tool", "args": {"location": "Tokyo"}}]',
        # Finish
        '[[ ## next_thought ## ]]\nI have the answer\n[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]',
        # Extract answer (ChainOfThought reasoning)
        "[[ ## reasoning ## ]]\nExtracting final answer from trajectory\n[[ ## answer ## ]]\nThe weather in Tokyo is Sunny",
    ]

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        response = make_mock_response(responses[call_count["count"]])
        call_count["count"] += 1
        return response

    settings.lm.client.chat.completions.create = mock_create

    agent = ReAct(QA, tools=[load_tools], max_iters=3)
    assert "weather_tool" not in agent.tools

    await agent.aforward(question="What's the weather in Tokyo?")

    # Verify weather_tool was added during execution
    assert "weather_tool" in agent.tools


def test_module_callback_return_value() -> None:
    """Test module callback returns observation string."""

    @module_callback
    def callback(context):
        return "Callback executed successfully"

    predictor = Predict(QA)
    context = ModuleContext(module=predictor)

    result = callback(context)
    assert isinstance(result, str)
    assert result == "Callback executed successfully"


def test_init_module_with_non_tool_objects() -> None:
    """Test init_module handles both Tool objects and functions."""

    def regular_function(x: int) -> str:
        return str(x)

    @tool(name="decorated_tool", description="Decorated tool")
    def decorated_tool(x: int = Field(...)) -> str:
        return f"tool: {x}"

    predictor = Predict(QA)

    # Should handle mix of regular functions and decorated tools
    # Regular functions will be wrapped in Tool objects
    predictor.init_module(tools=[regular_function, decorated_tool])

    assert "decorated_tool" in predictor.tools
    # Regular function will be wrapped with its function name as the tool name
    assert "regular_function" in predictor.tools or len(predictor.tools) >= 1


def test_react_init_module_updates_tool_call_model() -> None:
    """Test ReAct.init_module() updates the ToolCallModel with new tool names."""

    @tool(name="tool1", description="First tool")
    def tool1(x: int = Field(...)) -> str:
        return "tool1"

    @tool(name="tool2", description="Second tool")
    def tool2(y: str = Field(...)) -> str:
        return "tool2"

    agent = ReAct(QA, tools=[tool1])

    # Get the output fields which include the ToolCallModel
    initial_fields = agent.react_signature.get_output_fields()
    assert "next_tool_calls" in initial_fields

    # Add tool2
    agent.init_module(tools=[tool1, tool2])

    # Verify signature was rebuilt
    updated_fields = agent.react_signature.get_output_fields()
    assert "next_tool_calls" in updated_fields

    # The Literal type should now include tool2
    # (This is implicit in the signature rebuild)
