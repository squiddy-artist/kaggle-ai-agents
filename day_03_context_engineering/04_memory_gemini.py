import asyncio
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.genai import types

load_dotenv()
retry_config = types.HttpRetryOptions(attempts=4, exp_base=2, initial_delay=2)

async def run_session(runner: Runner, prompt: str, session_id: str):
    print(f"\nUser > {prompt}")
    query = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id="demo_user", session_id=session_id, new_message=query):
        if event.content and event.content.parts:
            print(f"Agent > {event.content.parts[0].text}")

async def main():
    # 1. Initialize Memory & Session
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    # 2. Create Agent
    user_agent = LlmAgent(
        model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
        name="MemoryDemoAgent",
        instruction="Answer user questions in simple words.",
    )

    # 3. Create Runner with BOTH
    runner = Runner(
        agent=user_agent,
        app_name="MemoryDemoApp",
        session_service=session_service,
        memory_service=memory_service,
    )

    # 4. Populate Session
    session_id = "conversation-01"
    await run_session(runner, "My favorite color is blue-green. Can you write a Haiku about it?", session_id)

    # 5. Verify Session
    session = await session_service.get_session(app_name="MemoryDemoApp", user_id="demo_user", session_id=session_id)
    print("\n✅ Session captured!")

    # 6. INGEST: Add to Memory
    await memory_service.add_session_to_memory(session)
    print("✅ Session added to long-term memory!")

if __name__ == "__main__":
    asyncio.run(main())