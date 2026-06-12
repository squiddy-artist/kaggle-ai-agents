import asyncio
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.memory import InMemoryMemoryService
from google.adk.tools import preload_memory
from google.genai import types


# ─────────────────────────────────────────────
# 📌 Constants — Centralized for easy changes
# ─────────────────────────────────────────────
APP_NAME  = "AutoMemoryLocalApp"
USER_ID   = "demo_user"
SESSION_1 = "local-session-1"
SESSION_2 = "local-session-2"


# ─────────────────────────────────────────────
# 💾 Callback: Auto-Save Session to Memory
# Runs automatically after every agent response
# Uses PUBLIC API (no underscore attributes)
# ─────────────────────────────────────────────
async def auto_save_to_memory(callback_context):
    print("  💾 [CALLBACK] Auto-saving session to memory...")
    try:
        await callback_context.memory_service.add_session_to_memory(
            callback_context.session
        )
        print("  ✅ [CALLBACK] Session successfully saved to memory.")
    except Exception as e:
        print(f"  ❌ [CALLBACK] Memory save failed: {e}")


# ─────────────────────────────────────────────
# 🏃 Run Session Helper
# Handles one full conversation turn
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
auto_memory_agent = LlmAgent(
    model=LiteLlm(model="ollama_chat/llama3.2"),
    name="AutoMemoryAgentLocal",
    instruction=(
        "You are a helpful assistant with long-term memory. "
        "At the start of EVERY conversation, you MUST call the preload_memory tool "
        "to retrieve relevant facts about the user from past sessions. "
        "Use those retrieved facts to give personalized, context-aware answers. "
        "Never rely on assumptions — always check memory first."
    ),
    tools=[preload_memory],
    after_agent_callback=auto_save_to_memory,
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    # --- Initialize Services ---
    memory_service  = InMemoryMemoryService()
    session_service = InMemorySessionService()

    auto_runner = Runner(
        agent=auto_memory_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
    )

    # --- Explicitly create both sessions ---
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_1
    )
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_2
    )

    # ─────────────────────────────────────────
    # 🧪 TEST: Cross-Session Memory Recall
    # ─────────────────────────────────────────
    print("\n" + "="*50)
    print("🧠 TEST: Cross-Session Memory Recall")
    print("="*50)

    # Session 1: Plant a fact → callback auto-saves to memory
    print("\n📌 Session 1: Teaching the agent a fact...")
    await run_session(auto_runner, "I love the color electric blue.", SESSION_1)

    # Session 2: New session → preload_memory pulls the fact back in
    print("\n🔍 Session 2: Testing if agent remembers...")
    await run_session(auto_runner, "What is my favorite color?", SESSION_2)

    print("\n✅ Workflow complete.")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
