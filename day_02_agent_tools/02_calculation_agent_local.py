import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, AgentTool
from google.genai import types

# 1. Define local helper tools
def get_fee_for_payment_method(method: str) -> float:
    return 0.02 if method == "Bank Transfer" else 0.05

def get_exchange_rate(currency: str) -> float:
    return 83.50  # Mock rate for USD to INR

# 2. Define a local calculator tool
# This replaces BuiltInCodeExecutor for local models like Ollama
def calculate_amount(amount: float, rate: float, fee: float) -> float:
    """Calculates final amount after fee and conversion."""
    return (amount * (1 - fee)) * rate

# 3. Define the Worker Agent (Calculation)
calculation_agent = Agent(
    name="CalculationAgent",
    model=LiteLlm(model="ollama_chat/llama3.2"),
    instruction="Use the calculate_amount tool to perform the final math.",
    tools=[FunctionTool(func=calculate_amount)],
)

# 4. Define the Orchestrator Agent
enhanced_currency_agent = Agent(
    name="enhanced_currency_agent",
    model=LiteLlm(model="ollama_chat/llama3.2"),
    instruction="Get fee, get rate, then use CalculationAgent to compute final amount.",
    tools=[
        FunctionTool(func=get_fee_for_payment_method), 
        FunctionTool(func=get_exchange_rate), 
        AgentTool(agent=calculation_agent)
    ],
)

# 5. Run the orchestration
async def main():
    runner = Runner(
        agent=enhanced_currency_agent, 
        session_service=InMemorySessionService(),
        app_name="currency_app"
    )
    
    # Create session
    await runner.session_service.create_session(
        app_name="currency_app", 
        user_id="demo", 
        session_id="currency-session"
    )

    print("🚀 Starting Orchestration...")
    query = types.Content(role="user", parts=[types.Part(text="Convert 1250 USD to INR via Bank Transfer.")])
    
    async for event in runner.run_async(user_id="demo", session_id="currency-session", new_message=query):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: 
                    print(f"Agent > {part.text}")
                if part.function_call: 
                    print(f"🛠️ Tool call: {part.function_call.name}")

if __name__ == "__main__":
    asyncio.run(main())