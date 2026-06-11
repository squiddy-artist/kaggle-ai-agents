import asyncio
from dotenv import load_dotenv

# Load local secrets
load_dotenv()

from google.adk.apps.app import App, EventsCompactionConfig
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService

# Using exp_base=2 to prevent long freezes if rate limited
retry_config = types.HttpRetryOptions(attempts=4, exp_base=2, initial_delay=2, http_status_codes=[429, 500, 503, 504])

# --- HELPER FUNCTION ---
async def run_session(runner: Runner, user_id: str, prompt: str, session_id: str, turn: int):
    """Simulates a single back-and-forth conversation turn."""
    print(f"\n[Turn {turn}] User > {prompt}")
    query_content = types.Content(role="user", parts=[types.Part(text=prompt)])
    
    response_text = ""
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=query_content):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_text += part.text
    
    print(f"[Turn {turn}] Agent > {response_text}")
    print("  [System] Pausing 5 seconds to respect API speed limits...")
    await asyncio.sleep(5) 


async def main():
    USER_ID = "nandakumar_dev"
    SESSION_ID = "compaction_demo"
    
    print("\n" + "#"*60)
    print("🗜️ CONTEXT COMPACTION (AUTO-SUMMARIZATION) DEMO")
    print("#"*60)
    
    chatbot_agent = Agent(
        # We are using the standard 'flash' model to avoid strict experimental quotas
        model=Gemini(model="gemini-2.5-flash", retry_options=retry_config),
        name="research_bot",
        description="A helpful research assistant",
    )
    
    # NEW: We wrap the agent in an App to enable Compaction logic
    research_app_compacting = App(
        name="research_app_compacting",
        root_agent=chatbot_agent,
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=3,  # Trigger a silent summary every 3 messages
            overlap_size=1,         # Keep the very last message in its original raw text format
        ),
    )

    db_url = "sqlite+aiosqlite:///my_agent_data.db" 
    session_service = DatabaseSessionService(db_url=db_url)
    
    research_runner_compacting = Runner(
        app=research_app_compacting, session_service=session_service
    )
    
    print("✅ App initialized with Events Compaction enabled (Trigger: 3 Turns).")
    
    # 1. Ensure the session exists in the DB
    try:
        await session_service.create_session(app_name="research_app_compacting", user_id=USER_ID, session_id=SESSION_ID)
    except Exception as e:
        if "already exists" in str(e).lower():
            pass # Safe to ignore
        else:
            raise e

    print("\n==================================================")
    print(f"🟢 STARTING RESEARCH SESSION: {SESSION_ID}")
    print("==================================================")
    
    # Run 4 consecutive turns. After Turn 3, the agent will silently summarize the past.
    await run_session(research_runner_compacting, USER_ID, "What is the latest news about AI in healthcare?", SESSION_ID, 1)
    await run_session(research_runner_compacting, USER_ID, "Are there any new developments in drug discovery?", SESSION_ID, 2)
    await run_session(research_runner_compacting, USER_ID, "Tell me more about the second development you found.", SESSION_ID, 3)
    await run_session(research_runner_compacting, USER_ID, "Who are the main companies involved in that?", SESSION_ID, 4)

    print("\n" + "#"*60)
    print("🕵️ INSPECTING DATABASE FOR THE SECRET SUMMARY")
    print("#"*60)

    # Pull the entire session history directly out of the database
    final_session = await session_service.get_session(
        app_name=research_runner_compacting.app_name,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    print("--- Scanning Database Events List ---")
    found_summary = False
    
    for idx, event in enumerate(final_session.events):
        # We are looking for the special 'compaction' attribute flag
        if getattr(event.actions, 'compaction', None):
            print("\n✅ SUCCESS! Found the Compaction Event hidden in the database!")
            print(f"  Author: {event.author}")
            print(f"  Compacted Data payload: {event}\n")
            found_summary = True
            break

    if not found_summary:
        print("\n❌ No compaction event found. The summarization failed to trigger.")

if __name__ == "__main__":
    asyncio.run(main())