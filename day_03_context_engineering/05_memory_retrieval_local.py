import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
# FIX 1: Correct import path for memory tools
from google.adk.tools import load_memory
from google.genai import types

async def run_session(runner: Runner, prompt: str, session_id: str):
    print(f"\nUser > {prompt}")
    query = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id="demo_user", session_id=session_id, new_message=query):
        if event.content and event.content.parts:
            # Check for text in response
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

async def main():
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    # Local Agent with load_memory tool
    user_agent = LlmAgent(
        model=LiteLlm(model="ollama_chat/llama3.2"),
        name="MemoryDemoAgentLocal",
        instruction="Answer user questions. Use load_memory tool if you need to recall past info.",
        tools=[load_memory],
    )

    runner = Runner(
        agent=user_agent, 
        app_name="MemoryDemoAppLocal", 
        session_service=session_service, 
        memory_service=memory_service
    )

    # FIX 2: Explicitly create sessions before running
    await session_service.create_session(
        app_name="MemoryDemoAppLocal", 
        user_id="demo_user", 
        session_id="color-session-01"
    )
    await session_service.create_session(
        app_name="MemoryDemoAppLocal", 
        user_id="demo_user", 
        session_id="color-session-02"
    )

    # Populate Memory
    await run_session(runner, "My favorite color is blue-green.", "color-session-01")
    
    # Retrieve the session we just ran and add it to memory
    session = await session_service.get_session(
        app_name="MemoryDemoAppLocal", 
        user_id="demo_user", 
        session_id="color-session-01"
    )
    await memory_service.add_session_to_memory(session)
    print("\n✅ Memory saved to In-Memory store.")

    # Test retrieval in NEW session
    await run_session(runner, "What is my favorite color?", "color-session-02")

if __name__ == "__main__":
    asyncio.run(main())