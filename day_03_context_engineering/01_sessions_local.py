import asyncio
import sqlite3
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.genai import types

async def run_session(runner: Runner, user_id: str, messages: list[str], session_id: str):
    print(f"\n{'='*50}\n🟢 STARTING SESSION: {session_id}\n{'='*50}")
    for msg in messages:
        print(f"\nUser > {msg}")
        query_content = types.Content(role="user", parts=[types.Part(text=msg)])
        response_text = ""
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=query_content):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: response_text += part.text
        print(f"Agent > {response_text}")

async def main():
    USER_ID = "nandakumar_dev"
    SESSION_ID = "test-db-session-01"
    APP_NAME = "db_app"
    
    # Persistent DB Agent
    db_agent = Agent(
        model=LiteLlm(model="ollama_chat/llama3.2"),
        name="db_bot",
        description="Local persistent agent",
    )
    
    db_session_service = DatabaseSessionService(db_url="sqlite+aiosqlite:///my_agent_data.db")
    db_runner = Runner(agent=db_agent, app_name=APP_NAME, session_service=db_session_service)
    
    # FIX: Check if session exists to prevent "already exists" errors
    try:
        await db_session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
        print(f"✅ Created new session: {SESSION_ID}")
    except Exception:
        print(f"ℹ️ Session {SESSION_ID} already exists, loading from disk...")
    
    await run_session(db_runner, USER_ID, ["I'm Sam. What is my name?", "What is the capital of France?"], SESSION_ID)
    print("\n✅ Session successfully persisted to SQLite database.")

if __name__ == "__main__":
    asyncio.run(main())