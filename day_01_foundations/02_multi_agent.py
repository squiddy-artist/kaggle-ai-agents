import os
import asyncio
from dotenv import load_dotenv

# 1. Load local secrets
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
# Notice we are importing AgentTool here!
from google.adk.tools import google_search, AgentTool
from google.genai import types

# 2. Configure retry options (Same safe config from single agent)
retry_config = types.HttpRetryOptions(
    attempts=5,  
    exp_base=7,  
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504] 
)

# 3. Define the Researcher Agent
research_agent = Agent(
    name="ResearchAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a specialized research agent. Your only job is to use the
    google_search tool to find 2-3 pieces of relevant information on the given topic and present the findings with citations.""",
    tools=[google_search],
    output_key="research_findings",  # Saves its output to the shared session memory
)

# 4. Define the Summarizer Agent
summarizer_agent = Agent(
    name="SummarizerAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Read the provided research findings: {research_findings}
    Create a concise summary as a bulleted list with 3-5 key points.""",
    output_key="final_summary",
)

# 5. Define the Boss (Root Coordinator)
root_agent = Agent(
    name="ResearchCoordinator",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""You are a research coordinator. Your goal is to answer the user's query by orchestrating a workflow.
    1. First, you MUST call the `ResearchAgent` tool to find relevant information on the topic provided by the user.
    2. Next, after receiving the research findings, you MUST call the `SummarizerAgent` tool to create a concise summary.
    3. Finally, present the final summary clearly to the user as your response.""",
    # We wrap the sub-agents in `AgentTool` to make them callable tools for the root agent.
    tools=[AgentTool(research_agent), AgentTool(summarizer_agent)],
)

# 6. Create the runner (We only give it the Root Boss agent)
runner = InMemoryRunner(agent=root_agent)

# 7. Asynchronous Local Execution Loop
async def main():
    print("✅ Multi-Agent Team Initialized. (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down team...")
            break
            
        print("\n[Coordinator is assigning tasks...]")
        
        # verbose=True lets you watch the Boss talk to the Researcher and Summarizer!
        await runner.run_debug(user_messages=user_input, verbose=True)

if __name__ == "__main__":
    asyncio.run(main())