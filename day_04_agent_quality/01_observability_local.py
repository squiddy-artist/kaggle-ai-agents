import asyncio
import logging
import uuid
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
from typing import List


# ─────────────────────────────────────────────
# 🪵 Observability: Configure Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    filename="agent_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("research_agent")


# ─────────────────────────────────────────────
# 🔧 Tool 1: Fetch Papers (NEW)
# Simulates fetching papers for a given topic
# ─────────────────────────────────────────────
def fetch_papers(topic: str) -> List[str]:
    """
    Fetches a list of paper titles related to the given topic.
    In production, replace this with a real API call (e.g., ArXiv, Semantic Scholar).
    """
    logger.debug(f"fetch_papers tool called with topic: {topic}")

    # Mock dataset — replace with real API in production
    mock_data = {
        "quantum computing": [
            "Quantum Supremacy Using a Programmable Superconducting Processor",
            "Variational Quantum Eigensolver: A New Paradigm",
            "Quantum Error Correction with Surface Codes",
            "Fault-Tolerant Quantum Computing with Ion Traps",
            "Photonic Quantum Computing: Recent Advances",
        ],
        "machine learning": [
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers",
            "GPT-4 Technical Report",
        ],
    }

    # Case-insensitive topic matching
    for key in mock_data:
        if key in topic.lower():
            papers = mock_data[key]
            logger.debug(f"fetch_papers returning {len(papers)} papers for topic '{topic}'")
            return papers

    # Default fallback if topic not found
    logger.warning(f"No papers found for topic: {topic}. Returning empty list.")
    return []


# ─────────────────────────────────────────────
# 🔧 Tool 2: Count Papers (ORIGINAL — kept)
# ─────────────────────────────────────────────
def count_papers(papers: List[str]) -> int:
    """
    Counts the number of papers in a list of strings.
    Always use this tool to count — never count manually.
    """
    logger.debug(f"count_papers tool received input: {papers}")
    count = len(papers)
    logger.debug(f"count_papers returning count: {count}")
    return count


# ─────────────────────────────────────────────
# 🤖 Agent Definition (IMPROVED instruction)
# ─────────────────────────────────────────────
research_agent = Agent(
    name="research_agent",
    model=LiteLlm(model="ollama_chat/llama3.2"),
    instruction=(
        "You are a research assistant. When asked to find or count papers on a topic, "
        "you MUST follow these steps in order:\n"
        "1. First call the fetch_papers tool with the topic to retrieve the list of papers.\n"
        "2. Then call the count_papers tool with that list to get the total count.\n"
        "3. Finally, respond with the paper titles and the total count.\n"
        "IMPORTANT: Always use both tools in sequence. Never fetch or count manually yourself."
    ),
    tools=[
        FunctionTool(func=fetch_papers),
        FunctionTool(func=count_papers)
    ]
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():
    # --- Session Setup ---
    session_service = InMemorySessionService()
    runner = Runner(
        agent=research_agent,
        session_service=session_service,
        app_name="obs_app"
    )

    # --- Generate unique IDs to avoid session collision ---
    user_id = "user_default"
    session_id = str(uuid.uuid4())
    logger.info(f"Starting session | user_id={user_id} | session_id={session_id}")

    # --- Explicitly create the session ---
    await session_service.create_session(
        app_name="obs_app",
        user_id=user_id,
        session_id=session_id
    )

    # --- Query now matches what the tools can actually do ---
    query_text = "Find the latest quantum computing papers and count them."
    new_message = types.Content(role="user", parts=[types.Part(text=query_text)])

    print(f"\n🚀 Running agent with query: {query_text}\n")
    logger.info(f"User query: {query_text}")

    # --- Run Agent with Error Handling (FIXED) ---
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        logger.info(f"Agent response: {part.text}")   # ✅ Log agent output
                        print(f"Agent > {part.text}")

    except Exception as e:
        logger.error(f"Agent run failed: {e}", exc_info=True)
        print(f"\n❌ Agent encountered an error: {e}")
        print("💡 Tip: Make sure Ollama is running — try: ollama serve")

    finally:
        logger.info("Session completed.")
        print("\n✅ Session complete. Check 'agent_debug.log' for full trace.")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
