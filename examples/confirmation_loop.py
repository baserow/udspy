"""Confirmation Loop Examples.

This example demonstrates the loop-based pattern for handling confirmations
with udspy agents. This pattern is ideal for:
- Interactive CLI applications
- Multiple confirmations per task
- Uniform confirmation handling

Run with: uv run python examples/confirmation_loop.py
"""

import asyncio

from udspy import (
    ConfirmationRejected,
    ConfirmationRequired,
    InputField,
    OutputField,
    ReAct,
    ResumeState,
    Signature,
    settings,
    tool,
)

# Configure settings (use your preferred model)
settings.configure(
    model="gpt-oss:20b-cloud",
    base_url="http://localhost:11434/v1",  # For local models
)


# Define tools with confirmation requirements
@tool(name="delete_file", description="Delete a file", require_confirmation=True)
def delete_file(path: str) -> str:
    """Delete a file - requires confirmation."""
    # In a real app, this would actually delete the file
    return f"File '{path}' has been deleted"


@tool(name="search", description="Search for information")
def search(query: str) -> str:
    """Search tool - no confirmation needed."""
    # Mock search results
    results = {
        "python": "Python is a high-level programming language",
        "file management": "File management involves creating, reading, updating, and deleting files",
    }

    for key, value in results.items():
        if key in query.lower():
            return f"Found: {value}"

    return f"No results for '{query}'"


# Define signature
class Task(Signature):
    """Complete a task using available tools."""

    task: str = InputField(description="The task to complete")
    result: str = OutputField(description="Result of the task")


async def example_basic_loop():
    """Example 1: Basic confirmation loop."""
    print("=== Example 1: Basic Confirmation Loop ===\n")

    agent = ReAct(Task, tools=[search, delete_file], max_iters=5)

    task = "Delete the file /tmp/old_cache.txt"
    resume_state = None

    while True:
        try:
            result = await agent.aforward(task=task, resume_state=resume_state)
            print("\n✓ Task completed!")
            print(f"  Result: {result.result}\n")
            break

        except ConfirmationRequired as e:
            print("\n⚠️  Confirmation Required")
            print(f"   Question: {e.question}")

            if e.tool_call:
                print(f"   Tool: {e.tool_call.name}")
                print(f"   Arguments: {e.tool_call.args}")

            # Simulate user input (in real app, use input())
            response = "yes"  # Auto-approve for demo
            print(f"\n   User responds: {response}")

            resume_state = ResumeState(e, response)

        except ConfirmationRejected as e:
            print(f"\n✗ Operation rejected: {e.message}\n")
            break


async def example_interactive_loop():
    """Example 2: Interactive loop with actual user input."""
    print("\n=== Example 2: Interactive Confirmation Loop ===\n")

    agent = ReAct(Task, tools=[search, delete_file], max_iters=5)

    task = input("Enter your task: ").strip()
    if not task:
        task = "Search for Python info and delete /tmp/test.txt"
        print(f"Using default task: {task}")

    resume_state = None
    confirmations = 0

    while confirmations < 5:  # Max 5 confirmation rounds
        try:
            result = await agent.aforward(task=task, resume_state=resume_state)
            print("\n✓ Task completed!")
            print(f"  Result: {result.result}\n")
            break

        except ConfirmationRequired as e:
            confirmations += 1
            print(f"\n⚠️  Confirmation Required ({confirmations}/5)")
            print(f"   {e.question}")

            if e.tool_call:
                print("\n   Tool Details:")
                print(f"   - Name: {e.tool_call.name}")
                print("   - Arguments:")
                for key, value in e.tool_call.args.items():
                    print(f"     {key}: {value}")

            print("\n   Options: yes, no, or edit (for JSON args)")
            response = input("   Your response: ").strip()

            if not response:
                response = "yes"  # Default to yes if empty

            resume_state = ResumeState(e, response)

        except ConfirmationRejected as e:
            print(f"\n✗ Operation rejected: {e.message}\n")
            break

    if confirmations >= 5:
        print("\n⚠️  Too many confirmation requests. Task aborted.\n")


async def example_with_helper():
    """Example 3: Using a helper function for cleaner code."""
    print("\n=== Example 3: Helper Function Pattern ===\n")

    async def run_with_confirmations(agent, max_confirmations=5, auto_approve_safe=True, **inputs):
        """Helper to run agent with confirmation handling.

        Args:
            agent: The agent to run
            max_confirmations: Max number of confirmation rounds
            auto_approve_safe: Whether to auto-approve safe operations
            **inputs: Input arguments for the agent

        Returns:
            The result prediction or None if rejected/failed
        """
        resume_state = None

        for attempt in range(max_confirmations):
            try:
                return await agent.aforward(resume_state=resume_state, **inputs)

            except ConfirmationRequired as e:
                print(f"\n⚠️  Confirmation {attempt + 1}/{max_confirmations}")
                print(f"   {e.question}")

                if e.tool_call:
                    print(f"   Tool: {e.tool_call.name}({e.tool_call.args})")

                # Auto-approve safe operations
                if auto_approve_safe and e.tool_call:
                    if e.tool_call.name in ["search", "read_file"]:
                        print("   ✓ Auto-approved (safe operation)")
                        response = "yes"
                    elif "/tmp/" in str(e.tool_call.args.get("path", "")):
                        print("   ✓ Auto-approved (safe path)")
                        response = "yes"
                    else:
                        response = "yes"  # For demo, approve all
                        print(f"   User approves: {response}")
                else:
                    response = "yes"  # For demo
                    print(f"   User approves: {response}")

                resume_state = ResumeState(e, response)

            except ConfirmationRejected as e:
                print(f"\n✗ Rejected: {e.message}")
                return None

        print(f"\n⚠️  Exceeded {max_confirmations} confirmations")
        return None

    # Use the helper
    agent = ReAct(Task, tools=[search, delete_file], max_iters=5)

    result = await run_with_confirmations(
        agent,
        task="Search for file management info and delete /tmp/cache.log",
        max_confirmations=5,
        auto_approve_safe=True,
    )

    if result:
        print(f"\n✓ Final result: {result.result}\n")


def example_sync_version():
    """Example 4: Sync version (no async/await)."""
    print("\n=== Example 4: Synchronous Version ===\n")

    agent = ReAct(Task, tools=[search, delete_file], max_iters=5)

    task = "Delete /tmp/example.txt"
    resume_state = None

    while True:
        try:
            # Note: Using .forward() or () instead of .aforward()
            result = agent(task=task, resume_state=resume_state)
            print(f"✓ Result: {result.result}\n")
            break

        except ConfirmationRequired as e:
            print(f"⚠️  {e.question}")
            if e.tool_call:
                print(f"   Tool: {e.tool_call.name}({e.tool_call.args})")

            response = "yes"  # Auto-approve for demo
            print(f"   Responding: {response}")
            resume_state = ResumeState(e, response)


async def main():
    # Example 1: Basic loop
    await example_basic_loop()

    # Example 2: Interactive (commented out to avoid blocking in demo)
    # await example_interactive_loop()

    # Example 3: Helper function
    await example_with_helper()


if __name__ == "__main__":
    asyncio.run(main())

    # Example 4: Sync version (run directly, without await)
    example_sync_version()

    print("\n✓ All examples completed!")
