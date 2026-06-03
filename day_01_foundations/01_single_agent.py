import os
import asyncio # Added to handle the asynchronous ADK methods locally
from dotenv import load_dotenv

# 1. Load local secrets
load_dotenv()

# 2. ADK Imports
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

# 3. Configure retry options 
retry_config = types.HttpRetryOptions(
    attempts=5,  
    exp_base=7,  
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504] 
)

# 4. Define your root agent
root_agent = Agent(
    name="helpful_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)

# 5. Create the runner
runner = InMemoryRunner(agent=root_agent)

# 6. Asynchronous Local Execution Loop
async def main():
    print("✅ Local Agent Initialized. (Type 'exit' to quit)")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down agent...")
            break
            
        print("Agent is thinking...")
        
        # Correct ADK call method for debugging and prototyping. 
        # Setting verbose=True lets you see the agent's internal thought logs!
        await runner.run_debug(user_messages=user_input, verbose=True)

if __name__ == "__main__":
    # Start the async loop required by the local environment
    asyncio.run(main())