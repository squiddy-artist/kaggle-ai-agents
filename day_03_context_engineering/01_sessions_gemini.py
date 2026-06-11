import os
import asyncio
import sqlite3
from dotenv import load_dotenv

# Load local secrets
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
retry_config = types.HttpRetryOptions(attempts=4, exp_base=2, initial_delay=2, http_status_codes=[429, 500, 503, 504])
#retry_config = types.HttpRetryOptions(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504])
#(old aggressive version)

# --- HELPER FUNCTION (Replacing Kaggle's hidden function) ---
async def run_session(runner: Runner, user_id: str, messages: list[str], session_id: str):
    """Simulates a back-and-forth conversation within a specific session."""
    print(f"\n{'='*50}")
    print(f"🟢 STARTING SESSION: {session_id}")
    print(f"{'='*50}")
    
    for msg in messages:
        print(f"\nUser > {msg}")
        query_content = types.Content(role="user", parts=[types.Part(text=msg)])
        
        response_text = ""
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=query_content):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text
        
        print(f"Agent > {response_text}")
        
        # --- THE FIX: Pacing the API calls ---
        print("  [System] Pausing 5 seconds to respect API speed limits...")
        await asyncio.sleep(5)

# --- DB INSPECTOR HELPER ---
def check_data_in_db():
    """Reads the SQLite database to prove the data is persisting on your hard drive."""
    print("\n🗄️ INSPECTING SQLITE DATABASE `my_agent_data.db`:")
    try:
        with sqlite3.connect("my_agent_data.db") as connection:
            cursor = connection.cursor()
            result = cursor.execute("SELECT app_name, session_id, author, content FROM events")
            print([_[0] for _ in result.description]) # Print column headers
            for each in result.fetchall():
                # Truncating the content string so it doesn't flood your terminal
                print(f"('{each[0]}', '{each[1]}', '{each[2]}', '{str(each[3])[:40]}...')")
    except sqlite3.OperationalError:
        print("Database not found or empty.")


async def main():
    USER_ID = "nandakumar_dev"
    
    print("\n" + "#"*60)
    print("🧠 PART 1: IN-MEMORY (RAM) SESSIONS")
    print("#"*60)
    
    ram_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="ram_bot",
        description="A text chatbot with temporary memory",
    )
    
    ram_session_service = InMemorySessionService()
    ram_runner = Runner(agent=ram_agent, app_name="ram_app", session_service=ram_session_service)
    
    # RAM gets wiped every time, so creating it here is always safe
    await ram_session_service.create_session(app_name="ram_app", user_id=USER_ID, session_id="ram-session-01")
    
    await run_session(
        ram_runner, USER_ID,
        ["Hi, I am Sam! What is the capital of United States?", "Hello! What is my name?"], 
        "ram-session-01"
    )

    print("\n" + "#"*60)
    print("💾 PART 2: PERSISTENT SQLITE DATABASE SESSIONS")
    print("#"*60)
    
    db_agent = Agent(
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="db_bot",
        description="A text chatbot with persistent memory",
    )
    
    db_url = "sqlite+aiosqlite:///my_agent_data.db" 
    db_session_service = DatabaseSessionService(db_url=db_url)
    db_runner = Runner(agent=db_agent, app_name="db_app", session_service=db_session_service)
    
    # --- THE FIX: Try to create, skip if it already exists ---
    try:
        await db_session_service.create_session(app_name="db_app", user_id=USER_ID, session_id="test-db-session-01")
    except Exception as e:
        if "already exists" in str(e).lower():
            print("  [System] Existing session found. Loading past memory...")
        else:
            raise e

    # THE ULTIMATE TEST: Notice we do NOT say "Hi I am Sam" here!
    await run_session(
        db_runner, USER_ID,
        [
            "I'm back! Do you still remember my name?", 
            "Great! What is the capital of France?"
        ], 
        "test-db-session-01"
    )
    
    try:
        await db_session_service.create_session(app_name="db_app", user_id=USER_ID, session_id="test-db-session-02")
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise e
            
    await run_session(
        db_runner, USER_ID,
        ["Hello! What is my name?"], 
        "test-db-session-02"
    )

    # Test 3: Inspect the physical database file
    check_data_in_db()

if __name__ == "__main__":
    asyncio.run(main())