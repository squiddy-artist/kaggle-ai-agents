import asyncio
import os
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import load_memory
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
APP_NAME  = "MemoryDemoApp"
USER_ID   = "demo_user"
SESSION_1 = "birthday-session-01"
SESSION_2 = "birthday-session-02"


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
        "You are a helpful assistant with long-term memory. "
        "When asked about anything the user has told you before, "
        "you MUST call the load_memory tool to search for relevant past information. "
        "Always use load_memory before answering questions about the user's preferences, "
        "personal details, or past conversations."
    ),
    tools=[load_memory],
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    # --- Initialize Services ---
    memory_service  = InMemoryMemoryService()
    session_service = InMemorySessionService()

    runner = Runner(
        agent=user_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
    )

    # --- Explicitly create both sessions (FIXED) ---
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_1
    )
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_2
    )

    # ─────────────────────────────────────────
    # 📌 Step 1: Populate Memory
    # ─────────────────────────────────────────
    print("\n" + "="*50)
    print("🧠 TEST: Manual Memory Retrieval")
    print("="*50)

    print("\n📌 Step 1: Saving a fact to memory...")
    await run_session(runner, "My birthday is on March 15th.", SESSION_1)

    # Safely retrieve session and save to memory
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_1
    )
    if session:
        await memory_service.add_session_to_memory(session)
        print("\n✅ Birthday saved to memory.")
    else:
        print("\n⚠️ Session not found — memory not saved.")

    # ─────────────────────────────────────────
    # 🔍 Step 2: Test Retrieval in New Session
    # ─────────────────────────────────────────
    print("\n🔍 Step 2: Testing recall in a new session...")
    await run_session(runner, "When is my birthday?", SESSION_2)

    # ─────────────────────────────────────────
    # 🔎 Step 3: Manual Memory Search
    # ─────────────────────────────────────────
    print("\n🔎 Step 3: Manual memory search...")
    try:
        search = await memory_service.search_memory(
            app_name=APP_NAME,
            user_id=USER_ID,
            query="birthday"
        )
        print(f"🔍 Manual search found {len(search.memories)} memories.")
    except Exception as e:
        print(f"❌ Memory search failed: {e}")

    print("\n✅ Workflow complete.")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
