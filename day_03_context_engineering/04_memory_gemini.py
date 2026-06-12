import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
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
APP_NAME   = "MemoryDemoApp"
USER_ID    = "demo_user"
SESSION_ID = "conversation-01"


# ─────────────────────────────────────────────
# 🔁 Retry Configuration
# Handles Gemini rate limits gracefully
# ─────────────────────────────────────────────
retry_config = types.HttpRetryOptions(
    attempts=4,
    exp_base=2,
    initial_delay=2,
    http_status_codes=[429, 500, 503, 504]
)


# ─────────────────────────────────────────────
# 🏃 Run Session Helper
# Handles one full conversation turn safely
# ─────────────────────────────────────────────
async def run_session(runner: Runner, prompt: str, session_id: str):
    print(f"\nUser > {prompt}")

    query = types.Content(role="user", parts=[types.Part(text=prompt)])

    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=query
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent > {part.text}")

    except Exception as e:
        print(f"\n❌ Session '{session_id}' failed: {e}")
        print("💡 Check your GOOGLE_API_KEY and internet connection.")


# ─────────────────────────────────────────────
# 🤖 Agent Definition
# ─────────────────────────────────────────────
user_agent = LlmAgent(
    model=Gemini(
        model="gemini-2.0-flash",
        api_key=API_KEY,
        retry_options=retry_config
    ),
    name="MemoryDemoAgent",
    instruction=(
        "You are a helpful assistant. "
        "Answer user questions clearly and in simple words. "
        "When writing creative content like poems or haikus, "
        "make them vivid and engaging."
    ),
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    # 1. Initialize Services
    memory_service  = InMemoryMemoryService()
    session_service = InMemorySessionService()

    # 2. Create Runner
    runner = Runner(
        agent=user_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
    )

    # 3. Explicitly Create Session (FIXED)
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    print("\n" + "="*50)
    print("🧠 DEMO: Session Ingestion into Memory")
    print("="*50)

    # 4. Run Conversation
    print("\n📌 Step 1: Running conversation...")
    await run_session(
        runner,
        "My favorite color is blue-green. Can you write a Haiku about it?",
        SESSION_ID
    )

    # 5. Safely Retrieve and Verify Session
    print("\n📌 Step 2: Verifying session...")
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    # 6. Safely Ingest Session into Memory
    print("\n📌 Step 3: Ingesting session into memory...")
    if session:
        try:
            await memory_service.add_session_to_memory(session)
            print("✅ Session added to long-term memory!")
        except Exception as e:
            print(f"❌ Memory ingest failed: {e}")
    else:
        print("⚠️ Session not found — memory not saved.")

    print("\n✅ Workflow complete.")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
