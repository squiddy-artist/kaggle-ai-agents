import os
import asyncio
from dotenv import load_dotenv

# 1. Load local secrets
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# --- THE TWO NEW IMPORTS FOR SECTION 3 ---
from google.adk.tools import AgentTool
from google.adk.code_executors import BuiltInCodeExecutor

# 2. Configure retry options 
retry_config = types.HttpRetryOptions(
    attempts=5,  
    exp_base=7,  
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504] 
)

# 3. Bring over your custom tools from the previous step
def get_fee_for_payment_method(method: str) -> dict:
    """Looks up the transaction fee percentage for a given payment method.
    Args: method: e.g., "platinum credit card" or "bank transfer".
    Returns: {"status": "success", "fee_percentage": 0.02}
    """
    fee_database = {"platinum credit card": 0.02, "gold debit card": 0.035, "bank transfer": 0.01}
    fee = fee_database.get(method.lower())
    if fee is not None: return {"status": "success", "fee_percentage": fee}
    return {"status": "error", "error_message": f"Payment method '{method}' not found"}

def get_exchange_rate(base_currency: str, target_currency: str) -> dict:
    """Looks up and returns the exchange rate between two currencies.
    Args: base_currency: e.g., "USD". target_currency: e.g., "EUR".
    Returns: {"status": "success", "rate": 0.93}
    """
    rate_database = {"usd": {"eur": 0.93, "jpy": 157.50, "inr": 83.58}}
    base, target = base_currency.lower(), target_currency.lower()
    rate = rate_database.get(base, {}).get(target)
    if rate is not None: return {"status": "success", "rate": rate}
    return {"status": "error", "error_message": f"Unsupported currency pair: {base_currency}/{target_currency}"}


# 4. NEW: Define the Calculation Specialist Agent
calculation_agent = Agent(
    name="CalculationAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a specialized calculator that ONLY responds with Python code. You are forbidden from providing any text, explanations, or conversational responses.
    Your task is to take a request for a calculation and translate it into a single block of Python code that calculates the answer.
     
    **RULES:**
    1. Your output MUST be ONLY a Python code block.
    2. Do NOT write any text before or after the code block.
    3. The Python code MUST calculate the result.
    4. The Python code MUST print the final result to stdout.
    5. You are PROHIBITED from performing the calculation yourself. Your only job is to generate the code that will perform the calculation.
    """,
    # This single line gives the agent access to a secure Python sandbox!
    code_executor=BuiltInCodeExecutor(), 
)

# 5. ENHANCED: The Main Currency Agent
enhanced_currency_agent = Agent(
    name="enhanced_currency_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a smart currency conversion assistant. You must strictly follow these steps and use the available tools.

  For any currency conversion request:
   1. Get Transaction Fee: Use the get_fee_for_payment_method() tool to determine the transaction fee.
   2. Get Exchange Rate: Use the get_exchange_rate() tool to get the currency conversion rate.
   3. Error Check: After each tool call, check the "status". If it's an error, stop and explain the issue.
   4. Calculate Final Amount (CRITICAL): You are strictly prohibited from performing any arithmetic calculations yourself. You must use the calculation_agent tool to generate Python code that calculates the final converted amount.
   5. Provide Detailed Breakdown: In your summary, you must state the final converted amount and explain how the result was calculated.
    """,
    tools=[
        get_fee_for_payment_method,
        get_exchange_rate,
        # Wrap the math agent so the main agent can call it like a tool!
        AgentTool(agent=calculation_agent), 
    ],
)

# 6. Create the runner
enhanced_runner = InMemoryRunner(agent=enhanced_currency_agent)

# 7. Execution Loop
async def main():
    print("✅ Enhanced Currency Agent Initialized. (Math Hallucinations Fixed!)")
    print("-" * 50)
    
    while True:
        user_input = input("\nEnter your conversion request: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down...")
            break
            
        if not user_input.strip():
            user_input = "Convert 1,250 USD to INR using a Bank Transfer. Show me the precise calculation."
            
        print("\n[Agent is pulling data and writing Python code to calculate...]")
        
        await enhanced_runner.run_debug(user_messages=user_input, verbose=True)

if __name__ == "__main__":
    asyncio.run(main())