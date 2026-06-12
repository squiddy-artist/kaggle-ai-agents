import os
import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types


# ─────────────────────────────────────────────
# 📌 Constants — Centralized for easy changes
# ─────────────────────────────────────────────
APP_NAME   = "db_app"
USER_ID    = "nandakumar_dev"
SESSION_ID = "test-db-session-01"
DB_URL     = "sqlite+aiosqlite:///my_agent_data.db"


# ─────────────────────────────────────────────
# 🏃 Run Session Helper
# Handles multi-turn conversation safely
# ─────────────────────────────────────────────
async def run_session(runner: Runner, user_id: str, messages: list[str], session_id: str):
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
            print("  💡 Make sure Ollama is running: ollama serve")
            continue

        print(f"[Turn {turn}] Agent > {response_text}")


# ─────────────────────────────────────────────
# 🤖 Agent Definition
# ─────────────────────────────────────────────
db_agent = Agent(
    model=LiteLlm(model="ollama_chat/llama3.2"),
    name="db_bot",
    description="Local persistent agent.",
    instruction=(
        "You are a helpful assistant with a persistent memory. "
        "Remember details the user shares and refer back to them naturally."
    ),
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    print("\n" + "#"*60)
    print("💾 DEMO: Persistent Sessions with SQLite (Local Ollama)")
    print("#"*60)
    print(f"  database = {DB_URL}")

    # --- Initialize Database Session Service ---
    db_session_service = DatabaseSessionService(db_url=DB_URL)

    db_runner = Runner(
        agent=db_agent,
        app_name=APP_NAME,
        session_service=db_session_service
    )

    # ─────────────────────────────────────────
    # 📌 Step 1: Create Session (Safe)
    # ─────────────────────────────────────────
    print("\n📌 Step 1: Creating session...")
    try:
        await db_session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        print(f"✅ Created new session: {SESSION_ID}")
    except Exception as e:
        err = str(e).lower()
        if "already exists" in err or "unique" in err:
            print(f"ℹ️ Session {SESSION_ID} already exists — reusing it.")
        else:
            raise  # Re-raise unexpected real errors

    # ─────────────────────────────────────────
    # 📌 Step 2: Multi-Turn Conversation
    # ─────────────────────────────────────────
    print("\n📌 Step 2: Running multi-turn conversation...")
    await run_session(
        db_runner,
        USER_ID,
        [
            "I'm Sam. What is my name?",
            "What is the capital of France?",
        ],
        SESSION_ID
    )

    # ─────────────────────────────────────────
    # 📌 Step 3: Verify Session Persisted
    # ─────────────────────────────────────────
    print("\n📌 Step 3: Verifying session persistence...")
    session = await db_session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    if session:
        print(f"\n🔍 Session verified in database!")
        print(f"  Session ID : {session.id}")
        print(f"  Events     : {len(session.events)} stored")
        print("\n✅ Session successfully persisted to SQLite database.")
    else:
        print("⚠️ Could not verify session in database.")

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

    print("\n✅ Demo complete.")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
