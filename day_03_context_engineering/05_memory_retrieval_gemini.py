import asyncio
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService, load_memory
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
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    # Agent with load_memory tool
    user_agent = LlmAgent(
        model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
        name="MemoryDemoAgent",
        instruction="Answer user questions. Use load_memory tool if you need to recall past info.",
        tools=[load_memory], 
    )

    runner = Runner(agent=user_agent, app_name="MemoryDemoApp", session_service=session_service, memory_service=memory_service)

    # 1. Populate Memory
    await run_session(runner, "My birthday is on March 15th.", "birthday-session-01")
    session = await session_service.get_session(app_name="MemoryDemoApp", user_id="demo_user", session_id="birthday-session-01")
    await memory_service.add_session_to_memory(session)
    print("✅ Birthday saved to memory.")

    # 2. Test Retrieval in NEW session
    await run_session(runner, "When is my birthday?", "birthday-session-02")

    # 3. Manual Search
    search = await memory_service.search_memory(app_name="MemoryDemoApp", user_id="demo_user", query="birthday")
    print(f"\n🔍 Manual search found {len(search.memories)} memories.")

if __name__ == "__main__":
    asyncio.run(main())