import os
import asyncio
import uuid
from dotenv import load_dotenv

# Load local secrets
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types

# Imports for Long-Running Operations
from google.adk.tools import ToolContext, FunctionTool
from google.adk.apps import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

retry_config = types.HttpRetryOptions(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504])

# --- 1. THE PAUSABLE TOOL ---
LARGE_ORDER_THRESHOLD = 5

def place_shipping_order(num_containers: int, destination: str, tool_context: ToolContext) -> dict:
    if num_containers <= LARGE_ORDER_THRESHOLD:
        return {"status": "approved", "message": f"Order auto-approved: {num_containers} containers to {destination}"}

    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Large order: {num_containers} containers to {destination}. Do you want to approve?",
            payload={"num_containers": num_containers, "destination": destination},
        )
        return {"status": "pending", "message": f"Order for {num_containers} containers requires approval"}

    if tool_context.tool_confirmation.confirmed:
        return {"status": "approved", "message": f"Order approved: {num_containers} containers to {destination}"}
    else:
        return {"status": "rejected", "message": f"Order rejected: {num_containers} containers to {destination}"}

# --- 2. ARCHITECTURE SETUP ---
shipping_agent = Agent(
    name="shipping_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a shipping coordinator assistant.
    1. Use the place_shipping_order tool.
    2. If the order status is 'pending', inform the user that approval is required.
    3. After receiving the final result, provide a clear summary.""",
    tools=[FunctionTool(func=place_shipping_order)],
)

# The "Memory Card"
shipping_app = App(
    name="shipping_coordinator",
    root_agent=shipping_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

session_service = InMemorySessionService()
shipping_runner = Runner(app=shipping_app, session_service=session_service)

# --- 3. WORKFLOW HELPER FUNCTIONS ---
def check_for_approval(events):
    """Check if events contain an approval request."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "adk_request_confirmation":
                    return {"approval_id": part.function_call.id, "invocation_id": event.invocation_id}
    return None

def print_agent_response(events):
    """Print agent's text responses from events."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"Agent > {part.text}")

def create_approval_response(approval_info, approved):
    """Translate human decision into ADK code."""
    confirmation_response = types.FunctionResponse(
        id=approval_info["approval_id"],
        name="adk_request_confirmation",
        response={"confirmed": approved},
    )
    return types.Content(role="user", parts=[types.Part(function_response=confirmation_response)])

# --- 4. THE MAIN EXECUTION LOOP ---
async def run_shipping_workflow(query: str, auto_approve: bool = True):
    print(f"\n{'='*60}")
    print(f"User > {query}\n")

    session_id = f"order_{uuid.uuid4().hex[:8]}"
    await session_service.create_session(app_name="shipping_coordinator", user_id="test_user", session_id=session_id)

    query_content = types.Content(role="user", parts=[types.Part(text=query)])
    events = []

    # STEP 1: Send initial request
    async for event in shipping_runner.run_async(user_id="test_user", session_id=session_id, new_message=query_content):
        events.append(event)

    # STEP 2: Detect Pause
    approval_info = check_for_approval(events)

    # STEP 3: Handle execution logic
    if approval_info:
        print(f"⏸️  Pausing for approval...")
        print(f"🤔 Human Decision: {'APPROVE ✅' if auto_approve else 'REJECT ❌'}\n")

        # RESUME the agent using the exact same invocation_id
        async for event in shipping_runner.run_async(
            user_id="test_user",
            session_id=session_id,
            new_message=create_approval_response(approval_info, auto_approve),
            invocation_id=approval_info["invocation_id"], 
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent > {part.text}")
    else:
        # No pause needed, print standard completion
        print_agent_response(events)

    print(f"{'='*60}\n")

# --- RUN THE KAGGLE DEMOS ---
async def main():
    # Demo 1: Small order (No pause)
    await run_shipping_workflow("Ship 3 containers to Singapore")

    # Demo 2: Large order -> Pauses -> Human Approves
    await run_shipping_workflow("Ship 10 containers to Rotterdam", auto_approve=True)

    # Demo 3: Large order -> Pauses -> Human Rejects
    await run_shipping_workflow("Ship 8 containers to Los Angeles", auto_approve=False)

if __name__ == "__main__":
    asyncio.run(main())