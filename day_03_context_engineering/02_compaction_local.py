import os
import asyncio
from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types


# ─────────────────────────────────────────────
# 📌 Constants — Centralized for easy changes
# ─────────────────────────────────────────────
APP_NAME   = "local_compaction_app"
USER_ID    = "nandakumar_dev"
SESSION_ID = "local-compaction-test"
DB_URL     = "sqlite+aiosqlite:///my_agent_data.db"


# ─────────────────────────────────────────────
# 🏃 Run Session Helper
# Handles one turn safely with full error handling
# ─────────────────────────────────────────────
async def run_session(runner: Runner, user_id: str, prompt: str, session_id: str, turn: int):
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
        print("  💡 Make sure Ollama is running: ollama serve")
        return

    if response_text:
        print(f"Agent > {response_text}")


# ─────────────────────────────────────────────
# 🤖 Agent Definition
# ─────────────────────────────────────────────
chatbot_agent = Agent(
    model=LiteLlm(model="ollama_chat/llama3.2"),
    name="research_bot",
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
research_app = App(
    name=APP_NAME,
    root_agent=chatbot_agent,
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=3,
        overlap_size=1
    ),
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    # --- Initialize Database Session Service ---
    db_service = DatabaseSessionService(db_url=DB_URL)
    runner = Runner(app=research_app, session_service=db_service)

    print("\n" + "#"*60)
    print("🗜️  DEMO: Event Compaction with Local Ollama")
    print("#"*60)
    print(f"  compaction_interval = 3 turns")
    print(f"  overlap_size        = 1 turn")
    print(f"  database            = {DB_URL}")

    # ─────────────────────────────────────────
    # 📌 Step 1: Create Session (Safe)
    # ─────────────────────────────────────────
    print("\n📌 Step 1: Creating session...")
    try:
        await db_service.create_session(
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
            raise

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

    final_session = await db_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
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
