import asyncio
from typing import Dict, Any
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types

def save_userinfo(tool_context: ToolContext, user_name: str, country: str) -> Dict[str, Any]:
    print(f"  🛠️ Saving to scratchpad: {user_name} from {country}")
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country
    return {"status": "success"}

def retrieve_userinfo(tool_context: ToolContext) -> Dict[str, Any]:
    return {
        "status": "success", 
        "user_name": tool_context.state.get("user:name", "Unknown"), 
        "country": tool_context.state.get("user:country", "Unknown")
    }

async def main():
    # Setup Services
    session_service = InMemorySessionService()
    
    root_agent = LlmAgent(
        model=LiteLlm(model="ollama_chat/llama3.2"),
        name="local_state_bot",
        tools=[save_userinfo, retrieve_userinfo],
        instruction="Use tools to save/retrieve user info. Call save_userinfo when name and country are provided."
    )

    runner = Runner(agent=root_agent, session_service=session_service, app_name="local_state_app")
    
    # FIX: Create the session explicitly
    session_id = "s1"
    await session_service.create_session(
        app_name="local_state_app", 
        user_id="user1", 
        session_id=session_id
    )
    
    # Run test
    print("\nUser > My name is Sam and I am from Poland.")
    query = types.Content(role="user", parts=[types.Part(text="My name is Sam and I am from Poland.")])
    async for event in runner.run_async(user_id="user1", session_id=session_id, new_message=query):
        pass 
    
    print("\nUser > Where am I from?")
    async for event in runner.run_async(user_id="user1", session_id=session_id, new_message=types.Content(role="user", parts=[types.Part(text="Where am I from?")])):
        if event.content and event.content.parts:
            # Print agent response if available
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

if __name__ == "__main__":
    asyncio.run(main())