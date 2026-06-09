import os
import asyncio
from dotenv import load_dotenv

# 1. Load local secrets
load_dotenv()

from google.adk.agents import Agent, SequentialAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# 2. Configure retry options for API stability
retry_config = types.HttpRetryOptions(
    attempts=5,  
    exp_base=7,  
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504] 
)

# 3. Agent 1: The Outliner
outline_agent = Agent(
    name="OutlineAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    instruction="""Create a highly engaging post outline for the given topic with:
    1. A catchy headline
    2. An introduction hook
    3. 3-5 main sections with 2-3 bullet points for each
    4. A strong call to action (encouraging comments or engagement)""",
    output_key="blog_outline",  # Passes data down the assembly line
)

# 4. Agent 2: The Writer
writer_agent = Agent(
    name="WriterAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # Injects the outline from the previous step
    instruction="""Following this outline strictly: {blog_outline}
    Write a brief, 200 to 300-word post with an engaging, professional, but approachable tone.""",
    output_key="blog_draft",  
)

# 5. Agent 3: The Editor
editor_agent = Agent(
    name="EditorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    # Injects the draft from the previous step
    instruction="""Edit this draft: {blog_draft}
    Your task is to polish the text by fixing any grammatical errors, improving the flow, ensuring the formatting is clean, and enhancing overall clarity for a social media audience.""",
    output_key="final_blog", 
)

# 6. The Assembly Line (SequentialAgent)
pipeline = SequentialAgent(
    name="ContentPipeline",
    # The order of this list is STRICT. It will always run exactly in this sequence.
    sub_agents=[outline_agent, writer_agent, editor_agent],
)

# 7. Create the runner
runner = InMemoryRunner(agent=pipeline)

# 8. Asynchronous Local Execution Loop
async def main():
    print("✅ Content Assembly Line Initialized. (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nEnter a topic for your post: ")
        
        if user_input.lower() in ['exit', 'quit']:
            print("Shutting down pipeline...")
            break
            
        print("\n[Pipeline is running... Outline -> Draft -> Edit]")
        
        # verbose=True will show the exact outputs passed between each agent
        await runner.run_debug(user_messages=user_input, verbose=True)

if __name__ == "__main__":
    asyncio.run(main())