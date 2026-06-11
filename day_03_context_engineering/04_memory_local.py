import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.genai import types

async def run_session(runner: Runner, prompt: str, session_id: str):
    print(f"\nUser > {prompt}")
    query = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id="demo_user", session_id=session_id, new_message=query):
        if event.content and event.content.parts:
            # Print agent response parts
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

async def main():
    # 1. Services
    memory_service = InMemoryMemoryService()
    session_service = InMemorySessionService()

    # 2. Local Agent (Using Ollama)
    user_agent = LlmAgent(
        model=LiteLlm(model="ollama_chat/llama3.2"),
        name="MemoryDemoAgentLocal",
        instruction="Answer user questions in simple words.",
    )

    runner = Runner(
        agent=user_agent,
        app_name="MemoryDemoAppLocal",
        session_service=session_service,
        memory_service=memory_service,
    )

    # 3. FIX: Explicitly create the session before running
    session_id = "conversation-local-01"
    await session_service.create_session(
        app_name="MemoryDemoAppLocal", 
        user_id="demo_user", 
        session_id=session_id
    )

    # 4. Execution
    await run_session(runner, "My favorite color is blue-green. Write a Haiku.", session_id)

    # 5. Ingest to Memory
    session = await session_service.get_session(
        app_name="MemoryDemoAppLocal", 
        user_id="demo_user", 
        session_id=session_id
    )
    
    await memory_service.add_session_to_memory(session)
    print("\n✅ Session successfully ingested into local memory store!")

if __name__ == "__main__":
    asyncio.run(main())