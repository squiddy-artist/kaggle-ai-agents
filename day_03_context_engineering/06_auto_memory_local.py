import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
# IMPORT FIX: preload_memory is located directly under google.adk.tools
from google.adk.tools import preload_memory
from google.genai import types

async def auto_save_to_memory(callback_context):
    print("  💾 [CALLBACK] Auto-saving local session...")
    # Access the memory service and current session from the invocation context
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

async def run_session(runner: Runner, prompt: str, session_id: str):
    print(f"\nUser > {prompt}")
    query = types.Content(role="user", parts=[types.Part(text=prompt)])
    
    # We collect the response to print it clearly
    async for event in runner.run_async(user_id="demo_user", session_id=session_id, new_message=query):
        if event.content and event.content.parts:
            # We print the text if the agent responds
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

async def main():
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    auto_memory_agent = LlmAgent(
        model=LiteLlm(model="ollama_chat/llama3.2"),
        name="AutoMemoryAgentLocal",
        instruction="Answer user questions. You have access to your long-term memory.",
        tools=[preload_memory],
        after_agent_callback=auto_save_to_memory,
    )

    auto_runner = Runner(
        agent=auto_memory_agent,
        app_name="AutoMemoryLocalApp",
        session_service=session_service,
        memory_service=memory_service,
    )

    # 1. Create sessions explicitly before running
    await session_service.create_session(
        app_name="AutoMemoryLocalApp", 
        user_id="demo_user", 
        session_id="local-session-1"
    )
    await session_service.create_session(
        app_name="AutoMemoryLocalApp", 
        user_id="demo_user", 
        session_id="local-session-2"
    )

    # 2. TEST: Cross-session recall
    # The callback will auto-save the first session to memory
    await run_session(auto_runner, "I love the color electric blue.", "local-session-1")
    
    # The preload_memory tool will automatically pull that fact into this new session
    await run_session(auto_runner, "What is my favorite color?", "local-session-2")
    
    print("\n✅ Workflow complete.")

if __name__ == "__main__":
    asyncio.run(main())