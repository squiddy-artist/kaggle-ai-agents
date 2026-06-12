import os
import asyncio
import sqlite3
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.genai import types


# ─────────────────────────────────────────────
# 🔐 Load & Validate API Key
# ─────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found. Check your .env file!")


# ─────────────────────────────────────────────
# 📌 Constants — Centralized for easy changes
# ─────────────────────────────────────────────
USER_ID      = "nandakumar_dev"
RAM_APP_NAME = "ram_app"
DB_APP_NAME  = "db_app"
RAM_SESSION  = "ram-session-01"
DB_SESSION_1 = "test-db-session-01"
DB_SESSION_2 = "test-db-session-02"
DB_URL       = "sqlite+aiosqlite:///my_agent_data.db"


# ─────────────────────────────────────────────
# 🔁 Retry Configuration
# exp_base=2 prevents long freezes if rate limited
# Old aggressive version kept as reference below
# ─────────────────────────────────────────────
retry_config = types.HttpRetryOptions(
    attempts=4,
    exp_base=2,
    initial_delay=2,
    http_status_codes=[429, 500, 503, 504]
)
# retry_config = types.HttpRetryOptions(
#     attempts=5, exp_base=7, initial_delay=1,
#     http_status_codes=[429, 500, 503, 504]
# )  # ← old aggressive version


# ─────────────────────────────────────────────
# 🏃 Run Session Helper
# Handles multi-turn conversation safely
# ─────────────────────────────────────────────
async def run_session(runner: Runner, user_id: str, messages: list[str], session_id: str):
    """Simulates a back-and-forth conversation within a specific session."""
    print(f"\n{'='*50}")
    print(f"🟢 STARTING SESSION: {session_id}")
    print(f"{'='*50}")

    for turn, msg in enumerate(messages, 1):
        print(f"\n[Turn {turn}] User > {msg}")
        query = types.Content(role="user", parts=[types.Part(text=msg)])

        response_text = ""
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=query
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            response_text += part.text

        except Exception as e:
            print(f"  ❌ Turn {turn} failed: {e}")
            print("  💡 Check your GOOGLE_API_KEY and internet connection.")
            continue

        print(f"[Turn {turn}] Agent > {response_text}")
        print("  [System] Pausing 5 seconds to respect API speed limits...")
        await asyncio.sleep(5)


# ─────────────────────────────────────────────
# 🗄️ DB Inspector Helper
# Reads raw SQLite to prove data is on disk
# ─────────────────────────────────────────────
def check_data_in_db():
    """Reads the SQLite database to prove data is persisting on your hard drive."""
    print("\n🗄️ INSPECTING SQLITE DATABASE `my_agent_data.db`:")
    try:
        with sqlite3.connect("my_agent_data.db") as connection:
            cursor = connection.cursor()

            # Check if events table exists first
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
            )
            if not cursor.fetchone():
                print("  ⚠️ Table 'events' not found — database may be empty or schema differs.")
                return

            result = cursor.execute(
                "SELECT app_name, session_id, author, content FROM events"
            )
            rows = result.fetchall()

            if not rows:
                print("  ⚠️ No rows found in events table.")
                return

            # Print column headers
            print(f"  Columns: {[col[0] for col in result.description]}")
            print(f"  Total rows: {len(rows)}\n")

            for row in rows:
                print(
                    f"  ('{row[0]}', '{row[1]}', '{row[2]}', "
                    f"'{str(row[3])[:40]}...')"
                )

    except sqlite3.OperationalError as e:
        print(f"  ❌ Database error: {e}")


# ─────────────────────────────────────────────
# 🤖 Agent Definitions
# ─────────────────────────────────────────────
ram_agent = Agent(
    model=Gemini(
        model="gemini-2.5-flash",
        api_key=API_KEY,
        retry_options=retry_config
    ),
    name="ram_bot",
    description="A text chatbot with temporary memory.",
    instruction=(
        "You are a friendly assistant with short-term memory. "
        "Remember what the user tells you during this conversation."
    ),
)

db_agent = Agent(
    model=Gemini(
        model="gemini-2.5-flash",
        api_key=API_KEY,
        retry_options=retry_config
    ),
    name="db_bot",
    description="A text chatbot with persistent memory.",
    instruction=(
        "You are a helpful assistant with persistent memory. "
        "Remember details the user shares and refer back to them naturally."
    ),
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    # ─────────────────────────────────────────
    # 🧠 PART 1: InMemory (RAM) Sessions
    # ─────────────────────────────────────────
    print("\n" + "#"*60)
    print("🧠 PART 1: IN-MEMORY (RAM) SESSIONS")
    print("#"*60)
    print("  ⚡ RAM sessions are wiped completely on restart.")

    ram_session_service = InMemorySessionService()
    ram_runner = Runner(
        agent=ram_agent,
        app_name=RAM_APP_NAME,
        session_service=ram_session_service
    )

    print("\n📌 Step 1: Creating RAM session (always fresh)...")
    await ram_session_service.create_session(
        app_name=RAM_APP_NAME,
        user_id=USER_ID,
        session_id=RAM_SESSION
    )
    print(f"✅ RAM session created: {RAM_SESSION}")

    print("\n📌 Step 2: Running RAM conversation...")
    await run_session(
        ram_runner, USER_ID,
        [
            "Hi, I am Sam! What is the capital of United States?",
            "Hello! What is my name?",
        ],
        RAM_SESSION
    )

    # ─────────────────────────────────────────
    # 💾 PART 2: Persistent SQLite DB Sessions
    # ─────────────────────────────────────────
    print("\n" + "#"*60)
    print("💾 PART 2: PERSISTENT SQLITE DATABASE SESSIONS")
    print("#"*60)
    print("  💡 DB sessions survive program restarts!")

    db_session_service = DatabaseSessionService(db_url=DB_URL)
    db_runner = Runner(
        agent=db_agent,
        app_name=DB_APP_NAME,
        session_service=db_session_service
    )

    # --- Session 1: Persistence Test ---
    print("\n📌 Step 3: Creating DB session 1 (persistence test)...")
    try:
        await db_session_service.create_session(
            app_name=DB_APP_NAME,
            user_id=USER_ID,
            session_id=DB_SESSION_1
        )
        print(f"✅ Created new session: {DB_SESSION_1}")
    except Exception as e:
        err = str(e).lower()
        if "already exists" in err or "unique" in err:
            print(f"ℹ️ Session {DB_SESSION_1} already exists — loading past memory...")
        else:
            raise

    print("\n📌 Step 4: Running DB session 1 conversation...")
    print("  (Notice: We do NOT say 'I am Sam' here — testing memory!)")
    await run_session(
        db_runner, USER_ID,
        [
            "I'm back! Do you still remember my name?",
            "Great! What is the capital of France?",
        ],
        DB_SESSION_1
    )

    # --- Session 2: Isolation Test ---
    print("\n📌 Step 5: Creating DB session 2 (isolation test)...")
    try:
        await db_session_service.create_session(
            app_name=DB_APP_NAME,
            user_id=USER_ID,
            session_id=DB_SESSION_2
        )
        print(f"✅ Created isolation session: {DB_SESSION_2}")
    except Exception as e:
        err = str(e).lower()
        if "already exists" in err or "unique" in err:
            print(f"ℹ️ Session {DB_SESSION_2} already exists — reusing it.")
        else:
            raise

    print("\n📌 Step 6: Running DB session 2 conversation...")
    print("  (This session should NOT know Sam's name!)")
    await run_session(
        db_runner, USER_ID,
        ["Hello! What is my name?"],
        DB_SESSION_2
    )

    # ─────────────────────────────────────────
    # 🗄️ PART 3: Raw SQLite Inspection
    # ─────────────────────────────────────────
    print("\n" + "#"*60)
    print("🗄️ PART 3: RAW SQLITE DATABASE INSPECTION")
    print("#"*60)
    print("  Proving data is physically on disk using raw SQL!")

    print("\n📌 Step 7: Inspecting database...")
    check_data_in_db()

    # ─────────────────────────────────────────
    # 🧹 Cleanup
    # ─────────────────────────────────────────
    print("\n" + "#"*60)
    print("🧹 CLEANUP: Removing database file")
    if os.path.exists("my_agent_data.db"):
        os.remove("my_agent_data.db")
        print("✅ Cleaned up my_agent_data.db successfully.")
    else:
        print("✅ No database file to clean up.")

    print("\n✅ All three parts complete.")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
