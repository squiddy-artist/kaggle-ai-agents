import asyncio
import logging
import uuid
import os
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
from typing import List


# ─────────────────────────────────────────────
# 🔐 Load Secrets from .env
# ─────────────────────────────────────────────
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found. Check your .env file!")


# ─────────────────────────────────────────────
# 🪵 Observability: Configure Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    filename="gemini_agent.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemini_research_agent")


# ─────────────────────────────────────────────
# 🔁 Retry Configuration
# Handles rate limits and transient API errors
# ─────────────────────────────────────────────
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=2,
    initial_delay=5,
    http_status_codes=[429, 500, 503, 504],
)


# ─────────────────────────────────────────────
# 🔧 Tool 1: Fetch Papers (NEW)
# Simulates fetching papers for a given topic
# Replace mock data with real API in production
# ─────────────────────────────────────────────
def fetch_papers(topic: str) -> List[str]:
    """
    Fetches a list of paper titles related to the given topic.
    In production, replace with a real API (e.g., ArXiv, Semantic Scholar).
    """
    logger.debug(f"fetch_papers called with topic: '{topic}'")

    # Mock dataset — replace with real API call in production
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
        "deep learning": [
            "Deep Residual Learning for Image Recognition",
            "Generative Adversarial Networks",
            "dropout: A Simple Way to Prevent Neural Networks from Overfitting",
        ],
    }

    # Case-insensitive topic matching
    for key in mock_data:
        if key in topic.lower():
            papers = mock_data[key]
            logger.debug(f"fetch_papers found {len(papers)} papers for '{topic}'")
            return papers

    # Fallback if topic not found
    logger.warning(f"No papers found for topic: '{topic}'. Returning empty list.")
    return []


# ─────────────────────────────────────────────
# 🔧 Tool 2: Count Papers (ORIGINAL — improved)
# ─────────────────────────────────────────────
def count_papers(papers: List[str]) -> int:
    """
    Counts the number of papers in a list of strings.
    Always use this tool to count — never count manually.
    """
    logger.debug(f"count_papers received: {papers}")
    count = len(papers)
    logger.debug(f"count_papers returning count: {count}")
    return count


# ─────────────────────────────────────────────
# 🤖 Agent Definition (IMPROVED instruction)
# ─────────────────────────────────────────────
research_agent = Agent(
    name="research_agent",
    model=Gemini(
        model="gemini-2.0-flash",
        api_key=API_KEY,
        retry_options=retry_config
    ),
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
        FunctionTool(func=count_papers),
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
        app_name="obs_gemini"
    )

    # --- Unique IDs to avoid session collision ---
    user_id = "user_default"
    session_id = str(uuid.uuid4())
    logger.info(f"Starting session | user_id={user_id} | session_id={session_id}")

    # --- Explicitly create the session (FIXED) ---
    await session_service.create_session(
        app_name="obs_gemini",
        user_id=user_id,
        session_id=session_id
    )

    # --- Query now matches what the tools can actually do ---
    query = "Find the latest quantum computing papers and count them."
    new_message = types.Content(role="user", parts=[types.Part(text=query)])

    print(f"\n🚀 Running Gemini Agent...")
    print(f"📝 Query: {query}\n")
    logger.info(f"User query: {query}")

    # --- Run Agent with full Error Handling (FIXED) ---
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        logger.info(f"Agent response: {part.text}")  # ✅ Log agent output
                        print(f"Agent > {part.text}")

    except Exception as e:
        logger.error(f"Agent run failed: {e}", exc_info=True)
        print(f"\n❌ Agent encountered an error: {e}")
        print("💡 Tips to fix:")
        print("   • Check your GOOGLE_API_KEY in .env is valid")
        print("   • Ensure you have Gemini API access enabled")
        print("   • Check your internet connection")

    finally:
        logger.info("Session completed.")
        print("\n✅ Session complete. Check 'gemini_agent.log' for full trace.")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
