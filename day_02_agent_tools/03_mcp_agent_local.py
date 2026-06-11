import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters
from google.genai import types

# 1. Initialize MCP Toolset
mcp_image_server = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="npx", 
            args=["-y", "@modelcontextprotocol/server-everything"],
            tool_filter=["getTinyImage"],
        )
    )
)

# 2. Agent Setup
image_agent = Agent(
    name="mcp_image_agent",
    model=LiteLlm(model="ollama_chat/llama3.2"),
    instruction="Use getTinyImage to generate an image. Explain what you are doing.",
    tools=[mcp_image_server],
)

async def main():
    # 3. Use the standardized Runner
    runner = Runner(
        agent=image_agent, 
        session_service=InMemorySessionService(),
        app_name="mcp_app"
    )
    
    # Session Initialization
    await runner.session_service.create_session(
        app_name="mcp_app",
        user_id="demo",
        session_id="mcp-session"
    )
    
    # 4. Execute using structured Content
    print("🚀 Starting MCP Agent...")
    
    message_content = types.Content(
        role="user", 
        parts=[types.Part(text="Provide a sample tiny image")]
    )
    
    # ... inside your async for loop ...
    async for event in runner.run_async(
        user_id="demo", 
        session_id="mcp-session", 
        new_message=message_content
    ):
        # 1. Print Text and Function Calls
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text: 
                    print(f"Agent > {part.text}")
                if part.function_call: 
                    print(f"🛠️ Calling: {part.function_call.name}")
        
        # 2. CORRECTED: Print Tool Results using get_function_responses()
        # This handles the internal Pydantic structure of the Event
        responses = event.get_function_responses()
        for response in responses:
            print(f"✅ Tool Result: {response.response}")

if __name__ == "__main__":
    asyncio.run(main())