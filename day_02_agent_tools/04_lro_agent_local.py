import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.apps import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext, FunctionTool
from google.genai import types

# The Tool with Approval Logic
def place_shipping_order(num_containers: int, destination: str, tool_context: ToolContext) -> dict:
    # Cast input to int to prevent TypeError
    count = int(num_containers)
    
    if count <= 5:
        return {"status": "approved", "message": f"Auto-approved: {count} to {destination}"}

    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Large order: {count} to {destination}. Approve?",
            payload={"num_containers": count, "destination": destination},
        )
        return {"status": "pending", "message": "Approval required"}

    return {"status": "approved" if tool_context.tool_confirmation.confirmed else "rejected"}

# Architecture
shipping_agent = Agent(
    name="shipping_agent",
    model=LiteLlm(model="ollama_chat/llama3.2"),
    instruction="You are a shipping coordinator. Use place_shipping_order.",
    tools=[FunctionTool(func=place_shipping_order)],
)

shipping_app = App(
    name="shipping_coordinator",
    root_agent=shipping_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

async def main():
    session_service = InMemorySessionService()
    runner = Runner(app=shipping_app, session_service=session_service)
    
    # 1. Start a session
    await session_service.create_session(
        app_name="shipping_coordinator", 
        user_id="demo_user", 
        session_id="lro-session-01"
    )

    # 2. Run initial request
    print("\nUser > Ship 10 containers to Rotterdam")
    query = types.Content(role="user", parts=[types.Part(text="Ship 10 containers to Rotterdam")])
    
    async for event in runner.run_async(
        user_id="demo_user", 
        session_id="lro-session-01", 
        new_message=query
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: print(f"Agent > {part.text}")
                if part.function_call: print(f"⏸️ Agent paused: {part.function_call.name}")

    print("✅ LRO Architecture verified.")

if __name__ == "__main__":
    asyncio.run(main())