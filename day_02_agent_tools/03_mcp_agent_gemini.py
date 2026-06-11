import os
import asyncio
import base64
from dotenv import load_dotenv

# 1. Load local secrets
load_dotenv()

from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types

# --- NEW IMPORTS FOR MCP ---
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

retry_config = types.HttpRetryOptions(attempts=5, exp_base=7, initial_delay=1, http_status_codes=[429, 500, 503, 504])

# 2. Define the MCP Toolset Connection
# This tells ADK to run the "server-everything" package using Node.js
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx",
            args=[
                "-y", 
                "@modelcontextprotocol/server-everything",
            ],
            tool_filter=["getTinyImage"], # We only want the image generator tool
        ),
        timeout=30,
    )
)

# 3. Create the Agent
image_agent = Agent(
    name="mcp_image_agent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""You are an assistant that provides test images. 
    If a user asks for an image, you MUST use the MCP Tool `getTinyImage` to generate it.
    Do NOT attempt to describe an image, just use the tool to fetch it.""",
    tools=[mcp_image_server], # Inject the MCP server here!
)

runner = InMemoryRunner(agent=image_agent)

# 4. Local Execution Loop with Image Saving
async def main():
    print("✅ MCP Image Agent Initialized. (Requires Node.js installed locally!)")
    print("-" * 50)
    
    # We remove the while loop so it only runs exactly once to protect your quota
    print("\n[Connecting to Node.js MCP Server via npx...]")
    
    # Run the agent
    response_events = await runner.run_debug("Provide a sample tiny image", verbose=True)
    
    image_saved = False
    for event in response_events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_response") and part.function_response:
                    content_list = part.function_response.response.get("content", [])
                    for item in content_list:
                        if item.get("type") == "image":
                            # Decode and save
                            image_data = base64.b64decode(item["data"])
                            file_path = "tiny_mcp_test.png"
                            with open(file_path, "wb") as f:
                                f.write(image_data)
                            print(f"\n🎉 SUCCESS! The MCP Server generated an image.")
                            print(f"📁 Image saved to: {file_path}")
                            image_saved = True
                            
                            # STOP THE AGENT IMMEDIATELY to prevent 429 quota errors on the final text response
                            return 
                            
    if not image_saved:
        print("\n❌ Could not extract image data. Make sure npx is installed.")

if __name__ == "__main__":
    asyncio.run(main())