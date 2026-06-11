import os
import asyncio
from dotenv import load_dotenv

# Load local secrets
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.genai import types

# The specific imports for Long-Running Operations
from google.adk.tools import ToolContext, FunctionTool
from google.adk.apps import App, ResumabilityConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

retry_config = types.HttpRetryOptions(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504])

# --- 1. THE TOOL WITH APPROVAL LOGIC ---
LARGE_ORDER_THRESHOLD = 5

def place_shipping_order(num_containers: int, destination: str, tool_context: ToolContext) -> dict:
    """Places a shipping order. Requires approval if ordering more than 5 containers."""
    
    # SCENARIO 1: Small orders (≤5) auto-approve
    if num_containers <= LARGE_ORDER_THRESHOLD:
        return {
            "status": "approved",
            "message": f"Order auto-approved: {num_containers} containers to {destination}",
        }

    # SCENARIO 2: First call, large order - PAUSE and ask for human approval
    if not tool_context.tool_confirmation:
        tool_context.request_confirmation(
            hint=f"⚠️ Large order: {num_containers} containers to {destination}. Do you want to approve?",
            payload={"num_containers": num_containers, "destination": destination},
        )
        return { 
            "status": "pending",
            "message": f"Order for {num_containers} containers requires approval",
        }

    # SCENARIO 3: Resumed call - Human has provided an answer
    if tool_context.tool_confirmation.confirmed:
        return {
            "status": "approved",
            "message": f"Order approved by human: {num_containers} containers to {destination}",
        }
    else:
        return {
            "status": "rejected",
            "message": f"Order rejected by human: {num_containers} containers to {destination}",
        }

# --- 2. THE AGENT ---
shipping_agent = Agent(
    name="shipping_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a shipping coordinator assistant.
    1. Use the place_shipping_order tool.
    2. If the order status is 'pending', inform the user that approval is required.
    3. After receiving the final result, provide a clear summary.""",
    tools=[FunctionTool(func=place_shipping_order)],
)

# --- 3. THE APP (Resumability / Save Game State) ---
shipping_app = App(
    name="shipping_coordinator",
    root_agent=shipping_agent,
    resumability_config=ResumabilityConfig(is_resumable=True),
)

# --- 4. THE RUNNER ---
session_service = InMemorySessionService()

shipping_runner = Runner(
    app=shipping_app,  
    session_service=session_service,
)

print("✅ LRO Setup Complete. (Waiting for workflow execution loop to be written!)")