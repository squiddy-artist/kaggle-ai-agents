import os
import asyncio
from dotenv import load_dotenv

# 1. Load local secrets
load_dotenv()

from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

# 2. Configure retry options 
retry_config = types.HttpRetryOptions(
    attempts=5,  
    exp_base=7,  
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504] 
)

# 3. Define the 3 Independent Researchers
tech_researcher = Agent(
    name="TechResearcher",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Research the latest AI/ML trends. Include 3 key developments, the main companies involved, and potential impact. Keep it under 100 words.",
    tools=[google_search],
    output_key="tech_research",
)

health_researcher = Agent(
    name="HealthResearcher",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Research recent medical breakthroughs. Include 3 advances, practical applications, and timelines. Keep it under 100 words.",
    tools=[google_search],
    output_key="health_research",
)

finance_researcher = Agent(
    name="FinanceResearcher",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="Research current fintech trends. Include 3 trends, market implications, and future outlook. Keep it under 100 words.",
    tools=[google_search],
    output_key="finance_research",
)

# 4. Define the Aggregator (The Executive Editor)
aggregator_agent = Agent(
    name="AggregatorAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    # Injects all three outputs simultaneously once they are ready
    instruction="""Combine these three research findings into a single executive summary:

    **Technology Trends:**
    {tech_research}
    
    **Health Breakthroughs:**
    {health_research}
    
    **Finance Innovations:**
    {finance_research}
    
    Your summary should highlight common themes, surprising connections, and the most important key takeaways from all three reports. The final summary should be around 200 words.""",
    output_key="executive_summary",
)

# 5. Build the Architecture
# A. Group the researchers into a team that fires at the same time
parallel_research_team = ParallelAgent(
    name="ParallelResearchTeam",
    sub_agents=[tech_researcher, health_researcher, finance_researcher],
)

# B. Tell the system to run the team FIRST, then run the aggregator SECOND
root_agent = SequentialAgent(
    name="ResearchSystem",
    sub_agents=[parallel_research_team, aggregator_agent],
)

# 6. Create the runner
runner = InMemoryRunner(agent=root_agent)

# 7. Asynchronous Local Execution Loop
async def main():
    print("✅ Parallel Executive Briefing System Initialized. (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nHit Enter to run the daily briefing (or type 'exit'): ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down system...")
            break
            
        print("\n[Dispatching researchers concurrently...]")
        
        # We pass a hardcoded prompt to trigger the whole system
        await runner.run_debug(
            user_messages="Run the daily executive briefing on Tech, Health, and Finance", 
            verbose=True
        )

if __name__ == "__main__":
    asyncio.run(main())