import os
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
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
APP_NAME  = "default"
USER_ID   = "nandakumar_dev"
SESSION_1 = "state-demo-session"
SESSION_2 = "new-isolated-session"


# ─────────────────────────────────────────────
# 🔁 Retry Configuration
# ─────────────────────────────────────────────
retry_config = types.HttpRetryOptions(
    attempts=4,
    exp_base=2,
    initial_delay=2,
    http_status_codes=[429, 500, 503, 504]
)


# ─────────────────────────────────────────────
# 🛠️ Tool: Save User Info to Session State
# ─────────────────────────────────────────────
def save_userinfo(tool_context: ToolContext, user_name: str, country: str) -> Dict[str, Any]:
    """
    Tool to record and save user name and country in session state.

    Args:
        user_name: The username to store in session state
        country: The name of the user's country
    """
    print(f"  🛠️ [TOOL EXECUTED] Saving to scratchpad -> Name: '{user_name}', Country: '{country}'")
    tool_context.state["user:name"]    = user_name
    tool_context.state["user:country"] = country
    return {"status": "success"}


# ─────────────────────────────────────────────
# 🛠️ Tool: Retrieve User Info from Session State
# ─────────────────────────────────────────────
def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user name and country from session state.
    """
    user_name = tool_context.state.get("user:name",    "Username not found")
    country   = tool_context.state.get("user:country", "Country not found")
    print(f"  🛠️ [TOOL EXECUTED] Retrieving from scratchpad -> Found: {user_name}, {country}")
    return {"status": "success", "user_name": user_name, "country": country}


# ─────────────────────────────────────────────
# 🏃 Run Session Helper
# Handles multi-turn conversations cleanly
# ─────────────────────────────────────────────
async def run_session(runner: Runner, user_id: str, messages: list, session_id: str):
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
# 🤖 Agent Definition (FIXED: added instruction)
# ─────────────────────────────────────────────
root_agent = LlmAgent(
    model=Gemini(
        model="gemini-2.0-flash",
        api_key=API_KEY,
        retry_options=retry_config
    ),
    name="text_chat_bot",
    description="A text chatbot with session state tools.",
    instruction=(
        "You are a helpful assistant. "
        "When the user provides their name and country, you MUST call save_userinfo to store it. "
        "When asked about the user's name or country, you MUST call retrieve_userinfo to fetch it. "
        "Never guess — always use the tools."
    ),
    tools=[save_userinfo, retrieve_userinfo],
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    # --- Initialize Services ---
    session_service = InMemorySessionService()

    runner = Runner(
        agent=root_agent,
        session_service=session_service,
        app_name=APP_NAME
    )

    print("\n" + "#"*60)
    print("🧰 SECTION 3: WORKING WITH SESSION STATE (CUSTOM TOOLS)")
    print("#"*60)
    print("✅ Agent with session state tools initialized!")

    # --- Explicitly create both sessions ---
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_1
    )
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_2
    )

    # ─────────────────────────────────────────
    # 🧪 TEST 1: Session State in Action
    # ─────────────────────────────────────────
    await run_session(
        runner, USER_ID,
        messages=[
            "Hi there, how are you doing today? What is my name?",  # Agent shouldn't know yet
            "My name is Sam. I'm from Poland.",                      # Agent should save it
            "What is my name? Which country am I from?",             # Agent should recall from state
        ],
        session_id=SESSION_1,
    )

    # ─────────────────────────────────────────
    # 🔍 Inspect Session State
    # ─────────────────────────────────────────
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_1
    )

    print("\n" + "="*50)
    print("🔍 INSPECTING SESSION STATE (state-demo-session)")
    print("="*50)
    if session:
        print(session.state)
        print("\nNotice the 'user:name' and 'user:country' keys storing our data!")
    else:
        print("⚠️ Session not found.")
    # ─────────────────────────────────────────────
    # 🧪 TEST 2: Session Isolation
    # ─────────────────────────────────────────────
    await run_session(
        runner, USER_ID,
        messages=["Hi there, how are you doing today? What is my name?"],
        session_id=SESSION_2,
    )

    session2 = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_2
    )

    print("\n" + "="*50)
    print("🔍 INSPECTING NEW SESSION STATE (new-isolated-session)")
    print("="*50)
    if session2:
        print(session2.state)
        print("\nBecause this is an isolated InMemorySession, the scratchpad is completely blank!")
    else:
        print("⚠️ Session not found.")

    # ─────────────────────────────────────────
    # 🧹 Cleanup
    # ─────────────────────────────────────────
    print("\n" + "#"*60)
    print("🧹 CLEANUP: Removing old databases")
    if os.path.exists("my_agent_data.db"):
        os.remove("my_agent_data.db")
        print("✅ Cleaned up old `my_agent_data.db` file.")
    else:
        print("✅ No database files to clean up.")

    print("\n✅ Workflow complete.")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
