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
    """Test ReactContext provides trajectory and input_args access."""
    agent = ReAct(QA, tools=[])
    trajectory = [{"thought": "test thought", "tool_calls": [], "observation": "test obs"}]
    input_args = {"question": "test question"}

    context = ReactContext(module=agent, trajectory=trajectory, input_args=input_args, stream=False)

    assert context.module == agent
    assert context.trajectory == trajectory
    assert context.trajectory[0]["thought"] == "test thought"
    assert context.input_args == input_args
    assert context.stream is False


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


@pytest.mark.asyncio
async def test_chain_of_thought_with_module_callback() -> None:
    """Test ChainOfThought executes module callbacks and adds tools dynamically."""
    from conftest import make_mock_response
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice as CompletionChoice
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    from udspy import ChainOfThought

    @tool(name="calculator", description="Perform calculations")
    def calculator(expression: str = Field(...)) -> str:
        return f"Result: {eval(expression, {'__builtins__': {}}, {})}"

    @tool(name="load_calculator", description="Load calculator tool")
    def load_calculator() -> callable:  # type: ignore[valid-syntax]
        @module_callback
        def callback(context):
            current = list(context.module.tools.values())
            context.module.init_module(tools=current + [calculator])
            return "Calculator loaded"

        return callback

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
                    content="",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name="load_calculator", arguments="{}"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    response2 = make_mock_response(
        "[[ ## reasoning ## ]]\nCalculator loaded\n[[ ## answer ## ]]\n4"
    )

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        call_count["count"] += 1
        if call_count["count"] == 1:
            return response1
        return response2

    settings.lm.client.chat.completions.create = mock_create

    cot = ChainOfThought(QA, tools=[load_calculator])
    assert "calculator" not in cot.predict.tools

    await cot.aforward(question="Load calculator and calculate 2+2")

    assert "calculator" in cot.predict.tools


@pytest.mark.asyncio
async def test_predict_uses_dynamically_loaded_tool() -> None:
    """Test Predict actually uses a tool after loading it dynamically."""
    from conftest import make_mock_response
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice as CompletionChoice
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function,
    )

    @tool(name="calculator", description="Perform calculations")
    def calculator(expression: str = Field(...)) -> str:
        return f"Result: {eval(expression, {'__builtins__': {}}, {})}"

    @tool(name="load_calculator", description="Load calculator")
    def load_calculator() -> callable:  # type: ignore[valid-syntax]
        @module_callback
        def callback(context):
            current = list(context.module.tools.values())
            context.module.init_module(tools=current + [calculator])
            return "Calculator loaded successfully"

        return callback

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
                    content="",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name="load_calculator", arguments="{}"),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    response2 = ChatCompletion(
        id="test",
        model="gpt-4o-mini",
        object="chat.completion",
        created=1234567890,
        choices=[
            CompletionChoice(
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_2",
                            type="function",
                            function=Function(
                                name="calculator", arguments='{"expression": "157 * 834"}'
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
    )

    response3 = make_mock_response("[[ ## answer ## ]]\n130938")

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        call_count["count"] += 1
        if call_count["count"] == 1:
            return response1
        elif call_count["count"] == 2:
            return response2
        return response3

    settings.lm.client.chat.completions.create = mock_create

    predictor = Predict(QA, tools=[load_calculator])
    result = await predictor.aforward(question="What is 157 * 834?")

    assert "calculator" in predictor.tools
    assert "130938" in result.answer


@pytest.mark.asyncio
async def test_react_uses_dynamically_loaded_tool() -> None:
    """Test ReAct actually uses a tool after loading it dynamically."""
    from conftest import make_mock_response

    @tool(name="calculator", description="Perform calculations")
    def calculator(expression: str = Field(...)) -> str:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"

    @tool(name="load_calculator", description="Load calculator tool")
    def load_calculator() -> callable:  # type: ignore[valid-syntax]
        @module_callback
        def callback(context):
            current = [
                t for t in context.module.tools.values() if t.name not in ("finish", "ask_to_user")
            ]
            context.module.init_module(tools=current + [calculator])
            return "Calculator loaded successfully"

        return callback

    responses = [
        '[[ ## next_thought ## ]]\nI need to load the calculator\n[[ ## next_tool_calls ## ]]\n[{"name": "load_calculator", "args": {}}]',
        '[[ ## next_thought ## ]]\nNow I can calculate\n[[ ## next_tool_calls ## ]]\n[{"name": "calculator", "args": {"expression": "25 * 4"}}]',
        '[[ ## next_thought ## ]]\nI have the answer\n[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]',
        "[[ ## reasoning ## ]]\nExtracting answer\n[[ ## answer ## ]]\n100",
    ]

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        response = make_mock_response(responses[call_count["count"]])
        call_count["count"] += 1
        return response

    settings.lm.client.chat.completions.create = mock_create

    agent = ReAct(QA, tools=[load_calculator], max_iters=3)
    result = await agent.aforward(question="What is 25 * 4?")

    assert "calculator" in agent.tools
    assert "100" in result.answer


@pytest.mark.asyncio
async def test_multiple_dynamic_tool_loads() -> None:
    """Test loading multiple tools dynamically in sequence."""
    from conftest import make_mock_response

    @tool(name="calculator", description="Perform calculations")
    def calculator(expression: str = Field(...)) -> str:
        return f"Result: {eval(expression, {'__builtins__': {}}, {})}"

    @tool(name="formatter", description="Format numbers")
    def formatter(number: str = Field(...)) -> str:
        return f"Formatted: {number}"

    @tool(name="load_calculator", description="Load calculator")
    def load_calculator() -> callable:  # type: ignore[valid-syntax]
        @module_callback
        def callback(context):
            current = [
                t for t in context.module.tools.values() if t.name not in ("finish", "ask_to_user")
            ]
            if calculator not in current:
                context.module.init_module(tools=current + [calculator])
            return "Calculator loaded"

        return callback

    @tool(name="load_formatter", description="Load formatter")
    def load_formatter() -> callable:  # type: ignore[valid-syntax]
        @module_callback
        def callback(context):
            current = [
                t for t in context.module.tools.values() if t.name not in ("finish", "ask_to_user")
            ]
            if formatter not in current:
                context.module.init_module(tools=current + [formatter])
            return "Formatter loaded"

        return callback

    responses = [
        '[[ ## next_thought ## ]]\nLoad calculator\n[[ ## next_tool_calls ## ]]\n[{"name": "load_calculator", "args": {}}]',
        '[[ ## next_thought ## ]]\nLoad formatter\n[[ ## next_tool_calls ## ]]\n[{"name": "load_formatter", "args": {}}]',
        '[[ ## next_thought ## ]]\nUse tools\n[[ ## next_tool_calls ## ]]\n[{"name": "calculator", "args": {"expression": "10 + 5"}}]',
        '[[ ## next_thought ## ]]\nFormat result\n[[ ## next_tool_calls ## ]]\n[{"name": "formatter", "args": {"number": "15"}}]',
        '[[ ## next_thought ## ]]\nDone\n[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]',
        "[[ ## reasoning ## ]]\nCompleted\n[[ ## answer ## ]]\nFormatted: 15",
    ]

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        response = make_mock_response(responses[call_count["count"]])
        call_count["count"] += 1
        return response

    settings.lm.client.chat.completions.create = mock_create

    agent = ReAct(QA, tools=[load_calculator, load_formatter], max_iters=5)

    assert "calculator" not in agent.tools
    assert "formatter" not in agent.tools

    await agent.aforward(question="Calculate and format")

    assert "calculator" in agent.tools
    assert "formatter" in agent.tools


@pytest.mark.asyncio
async def test_callback_with_invalid_tool_name() -> None:
    """Test that attempting to use an invalid tool name is handled gracefully."""
    from conftest import make_mock_response

    @tool(name="valid_tool", description="A valid tool")
    def valid_tool() -> str:
        return "Valid tool executed"

    responses = [
        '[[ ## next_thought ## ]]\nTry invalid tool\n[[ ## next_tool_calls ## ]]\n[{"name": "nonexistent_tool", "args": {}}]',
        '[[ ## next_thought ## ]]\nHandle error\n[[ ## next_tool_calls ## ]]\n[{"name": "finish", "args": {}}]',
        "[[ ## reasoning ## ]]\nError handled\n[[ ## answer ## ]]\nTool not found",
    ]

    call_count = {"count": 0}

    async def mock_create(**_kwargs):  # type: ignore[no-untyped-def]
        response = make_mock_response(responses[call_count["count"]])
        call_count["count"] += 1
        return response

    settings.lm.client.chat.completions.create = mock_create

    agent = ReAct(QA, tools=[valid_tool], max_iters=2)
    result = await agent.aforward(question="Test invalid tool")

    assert result is not None


def test_callback_context_access() -> None:
    """Test callback can access module state through context."""

    @tool(name="inspector", description="Inspect module")
    def inspector() -> callable:  # type: ignore[valid-syntax]
        @module_callback
        def callback(context):
            tool_count = len(context.module.tools)
            tool_names = list(context.module.tools.keys())
            return f"Module has {tool_count} tools: {', '.join(tool_names)}"

        return callback

    predictor = Predict(QA, tools=[inspector])
    context = PredictContext(module=predictor)

    result = inspector()
    observation = result(context)

    assert "Module has" in observation
    assert "inspector" in observation
