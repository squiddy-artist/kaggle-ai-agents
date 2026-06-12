import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.genai import types


# ─────────────────────────────────────────────
# 📌 Constants — Centralized for easy changes
# ─────────────────────────────────────────────
APP_NAME   = "MemoryDemoAppLocal"
USER_ID    = "demo_user"
SESSION_ID = "conversation-local-01"


# ─────────────────────────────────────────────
# 🏃 Run Session Helper
# Handles one full conversation turn safely
# ─────────────────────────────────────────────
async def run_session(runner: Runner, prompt: str, session_id: str):
    print(f"\nUser > {prompt}")

    query = types.Content(role="user", parts=[types.Part(text=prompt)])

    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=query
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(f"Agent > {part.text}")

    except Exception as e:
        print(f"\n❌ Session '{session_id}' failed: {e}")
        print("💡 Make sure Ollama is running: ollama serve")


# ─────────────────────────────────────────────
# 🤖 Agent Definition
# ─────────────────────────────────────────────
user_agent = LlmAgent(
    model=LiteLlm(model="ollama_chat/llama3.2"),
    name="MemoryDemoAgentLocal",
    instruction=(
        "You are a helpful assistant. "
        "Answer user questions clearly and in simple words. "
        "When writing creative content like poems or haikus, "
        "make them vivid and engaging."
    ),
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    # 1. Initialize Services
    memory_service  = InMemoryMemoryService()
    session_service = InMemorySessionService()

    # 2. Create Runner
    runner = Runner(
        agent=user_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
    )

    # 3. Explicitly Create Session
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    print("\n" + "="*50)
    print("🧠 DEMO: Session Ingestion into Memory")
    print("="*50)

    # 4. Run Conversation
    print("\n📌 Step 1: Running conversation...")
    await run_session(
        runner,
        "My favorite color is blue-green. Write a Haiku.",
        SESSION_ID
    )

    # 5. Safely Retrieve and Ingest Session into Memory
    print("\n📌 Step 2: Ingesting session into memory...")
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    if session:
        try:
            await memory_service.add_session_to_memory(session)
            print("\n✅ Session successfully ingested into local memory store!")
        except Exception as e:
            print(f"\n❌ Memory ingest failed: {e}")
    else:
        print("\n⚠️ Session not found — memory not saved.")

    print("\n✅ Workflow complete.")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
