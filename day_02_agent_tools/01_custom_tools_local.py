import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool # Required for tool wrapping
from google.genai import types

def get_fee_for_payment_method(method: str) -> dict:
    fee_database = {"platinum credit card": 0.02, "gold debit card": 0.035, "bank transfer": 0.01}
    fee = fee_database.get(method.lower())
    return {"status": "success", "fee_percentage": fee} if fee else {"status": "error", "error_message": "Not found"}

def get_exchange_rate(base: str, target: str) -> dict:
    rate_database = {"usd": {"eur": 0.93, "inr": 83.58}}
    rate = rate_database.get(base.lower(), {}).get(target.lower())
    return {"status": "success", "rate": rate} if rate else {"status": "error", "error_message": "Unsupported"}

currency_agent = Agent(
    name="currency_agent",
    model=LiteLlm(model="ollama_chat/llama3.2"),
    instruction="You are a currency assistant. Use the tools to calculate fees and conversions.",
    # Wrap functions in FunctionTool
    tools=[FunctionTool(func=get_fee_for_payment_method), FunctionTool(func=get_exchange_rate)],
)

async def main():
    # Use the standard Runner and SessionService
    runner = Runner(
        agent=currency_agent, 
        session_service=InMemorySessionService(),
        app_name="currency_tool_app"
    )
    
    print("🚀 Starting Currency Agent...")
    
    # Initialize a session for the agent to track history
    await runner.session_service.create_session(
        app_name="currency_tool_app", 
        user_id="demo", 
        session_id="currency-session"
    )
    
    query = types.Content(role="user", parts=[types.Part(text="I want to convert 500 US Dollars to Euros using my Platinum Credit Card.")])
    
    async for event in runner.run_async(user_id="demo", session_id="currency-session", new_message=query):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: print(f"Agent > {part.text}")

if __name__ == "__main__":
    asyncio.run(main())