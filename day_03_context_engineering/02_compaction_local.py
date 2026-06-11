import asyncio
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types

async def run_session(runner: Runner, user_id: str, prompt: str, session_id: str, turn: int):
    print(f"\n[Turn {turn}] User > {prompt}")
    query = types.Content(role="user", parts=[types.Part(text=prompt)])
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=query):
        if event.content and event.content.parts:
            # Safely print agent output
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

async def main():
    chatbot_agent = Agent(
        model=LiteLlm(model="ollama_chat/llama3.2"),
        name="research_bot",
    )
    
    research_app = App(
        name="local_compaction_app",
        root_agent=chatbot_agent,
        events_compaction_config=EventsCompactionConfig(compaction_interval=3, overlap_size=1),
    )

    # Database Service
    db_service = DatabaseSessionService(db_url="sqlite+aiosqlite:///my_agent_data.db")
    
    runner = Runner(app=research_app, session_service=db_service)
    
    # FIX: Explicitly create session for the database
    session_id = "local-compaction-test"
    try:
        await db_service.create_session(
            app_name="local_compaction_app", 
            user_id="nandakumar_dev", 
            session_id=session_id
        )
    except Exception as e:
        # If it already exists, that's fine
        pass

    # Execution Loop
    prompts = ["AI in health?", "Drug discovery?", "Tell me more.", "Companies?"]
    for i, prompt in enumerate(prompts, 1):
        await run_session(runner, "nandakumar_dev", prompt, session_id, i)

    print("\n✅ Compaction check complete. Database events processed.")

if __name__ == "__main__":
    asyncio.run(main())