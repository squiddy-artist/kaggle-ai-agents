import os
import asyncio
from dotenv import load_dotenv

# 1. Load local secrets
load_dotenv()

# Import the new LoopAgent and FunctionTool
from google.adk.agents import Agent, SequentialAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types

# 2. Configure retry options 
retry_config = types.HttpRetryOptions(
    attempts=5,  
    exp_base=7,  
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504] 
)

# 3. Define the Exit Tool (The emergency brake)
def exit_loop():
    """Call this function ONLY when the critique is 'APPROVED', indicating the story is finished and no more changes are needed."""
    return {"status": "approved", "message": "Story approved. Exiting refinement loop."}

# 4. Agent 1: The Initial Writer
initial_writer_agent = Agent(
    name="InitialWriterAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""Based on the user's prompt, write the first draft of a short story (around 100-150 words).
    Output only the story text, with no introduction or explanation.""",
    output_key="current_story",  # Stores the first draft
)

# 5. Agent 2: The Critic
critic_agent = Agent(
    name="CriticAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a constructive story critic. Review the story provided below.
    Story: {current_story}
    
    Evaluate the story's plot, characters, and pacing.
    - If the story is well-written and complete, you MUST respond with the exact phrase: "APPROVED"
    - Otherwise, provide 2-3 specific, actionable suggestions for improvement.""",
    output_key="critique",  # Stores the feedback
)

# 6. Agent 3: The Refiner (Has the power to exit)
refiner_agent = Agent(
    name="RefinerAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are a story refiner. You have a story draft and critique.
    
    Story Draft: {current_story}
    Critique: {critique}
    
    Your task is to analyze the critique.
    - IF the critique is EXACTLY "APPROVED", you MUST call the `exit_loop` function and nothing else.
    - OTHERWISE, rewrite the story draft to fully incorporate the feedback from the critique.""",
    output_key="current_story",  # Overwrites the old story with the new version
    tools=[FunctionTool(exit_loop)], # This gives the agent the ability to trigger the exit
)

# 7. Build the Architecture
# A. The Loop: Critic checks it, Refiner fixes it (Loops up to 2 times)
story_refinement_loop = LoopAgent(
    name="StoryRefinementLoop",
    sub_agents=[critic_agent, refiner_agent],
    max_iterations=2, 
)

# B. The Main Pipeline: Write Draft 1 -> Enter the Loop
root_agent = SequentialAgent(
    name="StoryPipeline",
    sub_agents=[initial_writer_agent, story_refinement_loop],
)

# 8. Create the runner
runner = InMemoryRunner(agent=root_agent)

# 9. Asynchronous Local Execution Loop
async def main():
    print("✅ Self-Refining Loop System Initialized. (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nEnter a topic for a story (or press Enter for the default): ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down system...")
            break
            
        if not user_input.strip():
            user_input = "Write a short story about a lighthouse keeper who discovers a mysterious, glowing map"
            
        print("\n[Writing initial draft and entering refinement loop...]")
        
        # verbose=True will show the critic tearing apart the first draft!
        await runner.run_debug(user_messages=user_input, verbose=True)

if __name__ == "__main__":
    asyncio.run(main())