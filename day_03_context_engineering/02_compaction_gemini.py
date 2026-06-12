import os
import asyncio
from dotenv import load_dotenv
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
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
APP_NAME   = "research_app_compacting"
USER_ID    = "nandakumar_dev"
SESSION_ID = "compaction_demo"
DB_URL     = "sqlite+aiosqlite:///my_agent_data.db"


# ─────────────────────────────────────────────
# 🔁 Retry Configuration
# exp_base=2 prevents long freezes if rate limited
# ─────────────────────────────────────────────
retry_config = types.HttpRetryOptions(
    attempts=4,
    exp_base=2,
    initial_delay=2,
    http_status_codes=[429, 500, 503, 504]
)


# ─────────────────────────────────────────────
# 🏃 Run Session Helper
# Handles one turn safely with full error handling
# ─────────────────────────────────────────────
async def run_session(runner: Runner, user_id: str, prompt: str, session_id: str, turn: int):
    """Simulates a single back-and-forth conversation turn."""
    print(f"\n[Turn {turn}] User > {prompt}")

    query = types.Content(role="user", parts=[types.Part(text=prompt)])

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
        return

    print(f"[Turn {turn}] Agent > {response_text}")
    print("  [System] Pausing 5 seconds to respect API speed limits...")
    await asyncio.sleep(5)


# ─────────────────────────────────────────────
# 🤖 Agent Definition (FIXED: added instruction)
# Using gemini-2.5-flash to avoid experimental quotas
# ─────────────────────────────────────────────
chatbot_agent = Agent(
    model=Gemini(
        model="gemini-2.5-flash",
        api_key=API_KEY,
        retry_options=retry_config
    ),
    name="research_bot",
    description="A helpful research assistant.",
    instruction=(
        "You are a concise research assistant specializing in AI and technology. "
        "Answer questions clearly and briefly. "
        "When asked follow-up questions, use context from earlier in the conversation."
    ),
)


# ─────────────────────────────────────────────
# 📦 App with Compaction Config
# Compacts every 3 turns, keeps 1 turn overlap
# ─────────────────────────────────────────────
research_app_compacting = App(
    name=APP_NAME,
    root_agent=chatbot_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,   # Trigger a silent summary every 3 messages
        overlap_size=1,          # Keep the last message in its original raw format
    ),
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    print("\n" + "#"*60)
    print("🗜️  CONTEXT COMPACTION (AUTO-SUMMARIZATION) DEMO")
    print("#"*60)
    print(f"  compaction_interval = 3 turns")
    print(f"  overlap_size        = 1 turn")
    print(f"  database            = {DB_URL}")

    # --- Initialize Database Session Service ---
    session_service = DatabaseSessionService(db_url=DB_URL)

    runner = Runner(
        app=research_app_compacting,
        session_service=session_service
    )

    print("\n✅ App initialized with Events Compaction enabled (Trigger: 3 Turns).")

    # ─────────────────────────────────────────
    # 📌 Step 1: Create Session (Safe)
    # ─────────────────────────────────────────
    print("\n📌 Step 1: Creating session...")
    try:
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=SESSION_ID
        )
        print("✅ New session created.")
    except Exception as e:
        err = str(e).lower()
        if "already exists" in err or "unique" in err:
            print("ℹ️ Session already exists — reusing it.")
        else:
            raise   # Re-raise unexpected real errors

    # ─────────────────────────────────────────
    # 📌 Step 2: Multi-Turn Conversation
    # ─────────────────────────────────────────
    print("\n📌 Step 2: Running multi-turn conversation...")
    print("\n" + "="*50)
    print(f"🟢 STARTING RESEARCH SESSION: {SESSION_ID}")
    print("="*50)

    prompts = [
        "What is the latest news about AI in healthcare?",
        "Are there any new developments in drug discovery?",
        "Tell me more about the second development you found.",
        "Who are the main companies involved in that?",
    ]

    for i, prompt in enumerate(prompts, 1):
        await run_session(runner, USER_ID, prompt, SESSION_ID, i)
        if i % 3 == 0:
            print(f"\n  🗜️ [Compaction should have fired after turn {i}]")

    # ─────────────────────────────────────────
    # 📌 Step 3: Inspect for Hidden Compaction Event
    # ─────────────────────────────────────────
    print("\n" + "#"*60)
    print("🕵️  INSPECTING DATABASE FOR THE SECRET SUMMARY")
    print("#"*60)

    final_session = await session_service.get_session(
        app_name=runner.app_name,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    if not final_session:
        print("⚠️ Could not retrieve session for inspection.")
    else:
        print(f"\n🔍 Total events stored after compaction: {len(final_session.events)}")
        print("  (Should be less than total turns — compaction summarized older turns!)")
        print("\n--- Scanning Database Events List ---")

        found_summary = False
        for idx, event in enumerate(final_session.events):
            if getattr(event.actions, 'compaction', None):
                print("\n✅ SUCCESS! Found the Compaction Event hidden in the database!")
                print(f"  Found at event index : {idx}")
                print(f"  Author               : {event.author}")
                print(f"  Compacted payload    : {event}\n")
                found_summary = True
                break

        if not found_summary:
            print("\n❌ No compaction event found.")
            print("  The summarization may not have triggered yet — try more turns.")

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

    print("\n✅ Compaction demo complete.")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
