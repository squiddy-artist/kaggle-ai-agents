import asyncio
import uuid
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.apps import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

# 1. The Tool (Updated with type casting)
def place_shipping_order(num_containers: int, destination: str, tool_context) -> dict:
    # Cast to int to handle cases where the LLM passes a string
    try:
        count = int(num_containers)
    except (ValueError, TypeError):
        return {"status": "error", "message": "Invalid container count provided."}

    if count <= 5:
        return {"status": "approved", "message": f"Order auto-approved: {count} containers to {destination}"}
    
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(hint="Approve order?", payload={"num": count})
        return {"status": "pending", "message": "Approval required"}
    
    return {"status": "approved" if tool_context.tool_confirmation.confirmed else "rejected"}

# 2. Setup
shipping_agent = Agent(
    name="shipping_agent",
    model=LiteLlm(model="ollama_chat/llama3.2"),
    instruction="You are a shipping coordinator. If status is pending, inform user.",
    tools=[FunctionTool(func=place_shipping_order)],
)

shipping_app = App(
    name="shipping_coordinator", 
    root_agent=shipping_agent, 
    resumability_config=ResumabilityConfig(is_resumable=True)
)

session_service = InMemorySessionService()
runner = Runner(app=shipping_app, session_service=session_service)

# 3. Workflow Logic
async def run_shipping_workflow(query: str, auto_approve: bool = True):
    print(f"\n{'='*50}\nUser > {query}")
    session_id = f"order_{uuid.uuid4().hex[:8]}"
    
    # Initialize session
    await session_service.create_session(
        app_name="shipping_coordinator", 
        user_id="test_user", 
        session_id=session_id
    )
    
    # First pass
    events = []
    async for event in runner.run_async(
        user_id="test_user", 
        session_id=session_id, 
        new_message=types.Content(role="user", parts=[types.Part(text=query)])
    ):
        events.append(event)
    
    # Detect if agent paused for approval
    approval_info = None
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    approval_info = {"id": part.function_call.id, "invocation_id": event.invocation_id}

    # Resume if paused
    if approval_info:
        print(f"⏸️ Pausing... Human Decision: {'APPROVE' if auto_approve else 'REJECT'}")
        resp = types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id=approval_info["id"], 
            name="adk_request_confirmation", 
            response={"confirmed": auto_approve}
        ))])
        
        async for event in runner.run_async(
            user_id="test_user", 
            session_id=session_id, 
            new_message=resp, 
            invocation_id=approval_info["invocation_id"]
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: print(f"Agent > {part.text}")
    else:
        # Standard flow
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text: print(f"Agent > {part.text}")

async def main():
    # Test auto-approval
    await run_shipping_workflow("Ship 3 containers to Singapore")
    # Test manual approval
    await run_shipping_workflow("Ship 10 containers to Rotterdam", auto_approve=True)

if __name__ == "__main__":
    asyncio.run(main())