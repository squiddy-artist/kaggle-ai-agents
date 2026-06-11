import os
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv

# Load local secrets
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext

# Using exp_base=2 to prevent long freezes if rate limited
retry_config = types.HttpRetryOptions(attempts=4, exp_base=2, initial_delay=2, http_status_codes=[429, 500, 503, 504])

# --- HELPER FUNCTION ---
async def run_session(runner: Runner, user_id: str, messages: list[str], session_id: str):
    """Simulates a back-and-forth conversation within a specific session."""
    print(f"\n{'='*50}")
    print(f"🟢 STARTING SESSION: {session_id}")
    print(f"{'='*50}")
    
    for turn, msg in enumerate(messages, 1):
        print(f"\n[Turn {turn}] User > {msg}")
        query_content = types.Content(role="user", parts=[types.Part(text=msg)])
        
        response_text = ""
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=query_content):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text
        
        print(f"[Turn {turn}] Agent > {response_text}")
        print("  [System] Pausing 5 seconds to respect API speed limits...")
        await asyncio.sleep(5) 

# --- 5.1 CUSTOM TOOLS FOR STATE MANAGEMENT ---

def save_userinfo(tool_context: ToolContext, user_name: str, country: str) -> Dict[str, Any]:
    """
    Tool to record and save user name and country in session state.

    Args:
        user_name: The username to store in session state
        country: The name of the user's country
    """
    print(f"  🛠️ [TOOL EXECUTED] Saving to scratchpad -> Name: '{user_name}', Country: '{country}'")
    # Write to session state using the 'user:' prefix for user data
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country
    return {"status": "success"}

def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Tool to retrieve user name and country from session state.
    """
    user_name = tool_context.state.get("user:name", "Username not found")
    country = tool_context.state.get("user:country", "Country not found")
    print(f"  🛠️ [TOOL EXECUTED] Retrieving from scratchpad -> Found: {user_name}, {country}")
    return {"status": "success", "user_name": user_name, "country": country}

async def main():
    APP_NAME = "default"
    USER_ID = "nandakumar_dev"

    print("\n" + "#"*60)
    print("🧰 SECTION 5: WORKING WITH SESSION STATE (CUSTOM TOOLS)")
    print("#"*60)

    # 5.2 Creating an Agent with Session State Tools
    root_agent = LlmAgent(
        model=Gemini(model="gemini-2.0-flash", retry_options=retry_config),
        name="text_chat_bot",
        description="""A text chatbot.
        Tools for managing user context:
        * To record username and country when provided use `save_userinfo` tool. 
        * To fetch username and country when required use `retrieve_userinfo` tool.
        """,
        tools=[save_userinfo, retrieve_userinfo],  # Provide the tools to the agent
    )

    # Set up session service and runner
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name=APP_NAME)
    
    print("✅ Agent with session state tools initialized!")
    
    # Safely create sessions
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id="state-demo-session")
    await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id="new-isolated-session")

    # 5.3 Testing Session State in Action
    await run_session(
        runner, USER_ID,
        [
            "Hi there, how are you doing today? What is my name?",  # Agent shouldn't know the name yet
            "My name is Sam. I'm from Poland.",                     # Provide name - agent should save it
            "What is my name? Which country am I from?",            # Agent should recall from session state
        ],
        "state-demo-session",
    )

    # 5.4 Inspecting Session State
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="state-demo-session"
    )
    
    print("\n" + "="*50)
    print("🔍 INSPECTING SESSION STATE (state-demo-session)")
    print("="*50)
    print(session.state)
    print("\nNotice the 'user:name' and 'user:country' keys storing our data!")

    # 5.5 Session State Isolation
    await run_session(
        runner, USER_ID,
        ["Hi there, how are you doing today? What is my name?"],
        "new-isolated-session",
    )

    # 5.6 Cross-Session State Sharing
    session2 = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id="new-isolated-session"
    )
    
    print("\n" + "="*50)
    print("🔍 INSPECTING NEW SESSION STATE (new-isolated-session)")
    print("="*50)
    print(session2.state)
    print("\nBecause this is an isolated InMemorySession, the scratchpad is completely blank!")

    # --- CLEANUP (From Notebook) ---
    print("\n" + "#"*60)
    print("🧹 CLEANUP: Removing old databases")
    if os.path.exists("my_agent_data.db"):
        os.remove("my_agent_data.db")
        print("✅ Cleaned up old `my_agent_data.db` file from previous exercises.")
    else:
        print("✅ No database files to clean up.")

if __name__ == "__main__":
    asyncio.run(main())