import asyncio
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService, preload_memory
from google.genai import types

load_dotenv()
retry_config = types.HttpRetryOptions(attempts=4, exp_base=2, initial_delay=2)

# The Automatic Save Callback
async def auto_save_to_memory(callback_context):
    """Automatically save session to memory after each agent turn."""
    print("  💾 [CALLBACK] Auto-saving session to memory...")
    await callback_context._invocation_context.memory_service.add_session_to_memory(
        callback_context._invocation_context.session
    )

async def run_session(runner: Runner, prompt: str, session_id: str):
    print(f"\nUser > {prompt}")
    query = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id="demo_user", session_id=session_id, new_message=query):
        if event.content and event.content.parts:
            print(f"Agent > {event.content.parts[0].text}")

async def main():
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    # Create Agent with automatic saving
    auto_memory_agent = LlmAgent(
        model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
        name="AutoMemoryAgent",
        instruction="Answer user questions using your memory.",
        tools=[preload_memory], # Proactive retrieval
        after_agent_callback=auto_save_to_memory, # The magic "checkpoint"
    )

    auto_runner = Runner(
        agent=auto_memory_agent,
        app_name="AutoMemoryApp",
        session_service=session_service,
        memory_service=memory_service,
    )

    # TEST: Cross-session recall
    await run_session(auto_runner, "I gifted a new toy to my nephew on his 1st birthday!", "auto-save-1")
    await run_session(auto_runner, "What did I gift my nephew?", "auto-save-2") # Different session!

if __name__ == "__main__":
    asyncio.run(main())