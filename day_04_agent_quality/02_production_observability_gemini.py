import asyncio
import logging
import os
import sys
import traceback
import warnings
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.plugins.logging_plugin import LoggingPlugin
from google.genai import types
from typing import List


# ─────────────────────────────────────────────
# 🪵 Structured Logging Setup
# Philosophy:
#   Terminal  → clean INFO summary   (what matters to YOU)
#   File      → full DEBUG trace     (what matters when debugging)
#   3rd party → CRITICAL only        (total silence on their internals)
#
# To debug: set LOG_LEVEL=DEBUG in your .env
# .env rule: NO inline comments after values
# ─────────────────────────────────────────────

load_dotenv()  # must be first — LOG_LEVEL needed before logging setup

# Defensive parse — strips inline comments if they sneak into .env
_raw_level = os.getenv("LOG_LEVEL", "INFO").split("#")[0].strip().upper()
LOG_LEVEL   = _raw_level if _raw_level in {"DEBUG", "INFO", "WARNING", "ERROR"} else "INFO"

# ── Root logger: WARNING baseline — silence everything by default ──
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ── Your named app logger ─────────────────────────────────────────
logger = logging.getLogger("agent_prod")
logger.setLevel(getattr(logging, LOG_LEVEL))
logger.propagate = False  # isolated — doesn't bubble to root

# Console handler: timestamp + level + message only
_console = logging.StreamHandler(sys.stdout)
_console.setLevel(getattr(logging, LOG_LEVEL))
_console.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S"
))
logger.addHandler(_console)

# File handler: full DEBUG always — flight recorder
_file = logging.FileHandler("agent_prod.log", mode="a", encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s | %(funcName)s:%(lineno)d | %(message)s"
))
logger.addHandler(_file)

# ── Silence ALL ADK + 3rd party internals at CRITICAL ─────────────
# Must use CRITICAL (not WARNING) — ADK logs full tracebacks at ERROR
for _noisy in [
    "LiteLLM", "httpx", "httpcore", "openai",
    "google_adk",
    "google.adk",
    "google_adk.google.adk.workflow._node_runner",
    "google_adk.google.adk.runners",
    "google_adk.google.adk.flows",
    "google_adk.google.adk.models",
    "google_adk.google.adk.plugins.plugin_manager",
]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# ── Silence ADK experimental feature UserWarnings ─────────────────
# Suppresses: "UserWarning: [EXPERIMENTAL] feature JSON_SCHEMA_FOR_FUNC_DECL"
# This comes from function_tool.py and cannot be stopped via logging
warnings.filterwarnings(
    "ignore",
    message=".*EXPERIMENTAL.*",
    category=UserWarning,
    module="google.*"
)


# ─────────────────────────────────────────────
# 🔑 API Key Validation
# ─────────────────────────────────────────────
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found — add it to your .env file!")


# ─────────────────────────────────────────────
# 📌 Constants
# ─────────────────────────────────────────────
APP_NAME   = "prod_agent"
USER_ID    = "nandakumar_dev"
SESSION_ID = "gemini-obs-session-01"


# ─────────────────────────────────────────────
# 🔌 Custom Plugin — ToolCounterPlugin
# Key names confirmed via live DEBUG run:
#   ADK plugin_manager passes: 'tool', 'tool_args', 'tool_context', 'result'
#   Same internal keys for ALL backends (Gemini, Ollama, etc.)
# ─────────────────────────────────────────────
class ToolCounterPlugin(BasePlugin):
    """Custom production monitoring plugin. Tracks every tool call."""

    def __init__(self):
        super().__init__(name="tool_counter")
        self.count = 0

    async def after_tool_callback(self, **kwargs):
        """
        Fires after every tool execution.
        **kwargs is version-resilient.
        """
        self.count += 1

        tool      = kwargs.get('tool')
        tool_name = getattr(tool, 'name', 'unknown')
        args      = kwargs.get('tool_args', {})   # confirmed: not 'args'
        response  = kwargs.get('result', 'N/A')   # confirmed: not 'tool_response'

        # Terminal: one clean line per tool call
        logger.info(
            "🔧 Tool fired: '%s' | args: %s | result: %s | total_calls: %d",
            tool_name, args, response, self.count
        )
        # File only: full kwargs keys for future debugging
        logger.debug("Tool kwargs keys: %s", list(kwargs.keys()))


# ─────────────────────────────────────────────
# 🛠️ Tool Definition
# ─────────────────────────────────────────────
def count_papers(papers: List[str]) -> int:
    """
    Counts the number of research papers in a given list.

    Args:
        papers: A list of paper titles or identifiers to count.

    Returns:
        The total number of papers as an integer.
    """
    result = len(papers) if isinstance(papers, list) else 0
    logger.info("📄 count_papers → %d items → result: %d", len(papers), result)
    logger.debug("Papers received: %s", papers)  # file only — can be long
    return result


# ─────────────────────────────────────────────
# 🤖 Agent Definition
# ─────────────────────────────────────────────
research_agent = Agent(
    name="research_agent",
    model=Gemini(model="gemini-2.0-flash", api_key=API_KEY),
    instruction=(
        "You are a research assistant. When given a list of papers, "
        "use the count_papers tool to count them and report the result clearly."
    ),
    tools=[FunctionTool(func=count_papers)]
)


# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    print("\n" + "#"*60)
    print("🔌 DEMO: Dual Plugin Observability (Gemini)")
    print(f"   Log Level : {LOG_LEVEL}")
    print(f"   Log File  : agent_prod.log  ← full DEBUG trace always saved")
    print("#"*60)

    # ── Step 1: Session ───────────────────────
    logger.info("📌 Initializing session: %s", SESSION_ID)
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    logger.info("✅ Session ready")

    # ── Step 2: Runner — plugins controlled by LOG_LEVEL ─────────
    # LoggingPlugin uses print() internally — bypasses logging system
    # So we only attach it in DEBUG mode where verbosity is expected
    logger.info("📌 Building runner...")
    plugin = ToolCounterPlugin()

    active_plugins = [plugin]                        # always: your monitor
    if LOG_LEVEL == "DEBUG":
        active_plugins.insert(0, LoggingPlugin())    # dev only: ADK lifecycle logs

    runner = Runner(
        agent=research_agent,
        session_service=session_service,
        app_name=APP_NAME,
        plugins=active_plugins
    )
    logger.info(
        "✅ Runner ready — plugins: %s",
        ["LoggingPlugin", "ToolCounterPlugin"] if LOG_LEVEL == "DEBUG"
        else ["ToolCounterPlugin"]
    )

    # ── Step 3: Run Agent ─────────────────────
    logger.info("📌 Running Gemini agent...")
    print("─" * 60)

    query = types.Content(
        role="user",
        parts=[types.Part(
            text=(
                "Count these papers: "
                "['Quantum Supremacy 2024', 'QEC Advances', 'Topological Qubits Review']"
            )
        )]
    )

    response_text = ""
    try:
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=SESSION_ID,
            new_message=query
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text += part.text

    except KeyboardInterrupt:
        logger.warning("⚠️  Run interrupted by user.")
        return

    except Exception as e:
        error_type = type(e).__name__
        error_msg  = str(e)

        # ── 429 Rate Limit: dedicated clean block ─────────────────
        if "ResourceExhausted" in error_type or "429" in error_msg[:60]:
            print(f"\n{'─'*60}")
            print(f"⏳  RATE LIMIT HIT (429 RESOURCE_EXHAUSTED)")
            print(f"    Model  : gemini-2.0-flash")
            print(f"    Reason : Free tier quota exhausted")
            print(f"    Fix 1  : Wait ~60s and retry")
            print(f"    Fix 2  : Switch to gemini-1.5-flash (separate quota)")
            print(f"    Fix 3  : Enable billing at https://ai.dev/rate-limit")
            print(f"{'─'*60}")
            logger.warning(
                "429 quota exhausted on gemini-2.0-flash — "
                "full error saved to error_trace.log"
            )
            with open("error_trace.log", "w", encoding="utf-8") as f:
                f.write("429 RESOURCE_EXHAUSTED\n\n")
                traceback.print_exc(file=f)
            return

        # ── All other errors: minimal terminal + full file ─────────
        print(f"\n{'─'*60}")
        print(f"❌  AGENT RUN FAILED")
        print(f"    Type    : {error_type}")
        print(f"    Message : {error_msg[:120]}{'...' if len(error_msg) > 120 else ''}")
        tb_lines = traceback.format_tb(e.__traceback__)
        print(f"    Where   :")
        for line in tb_lines[-3:]:
            print(f"      {line.strip()}")
        print(f"{'─'*60}")
        with open("error_trace.log", "w", encoding="utf-8") as f:
            f.write(f"ERROR: {error_type}: {e}\n\n")
            traceback.print_exc(file=f)
        logger.error("Run failed: %s — full trace → error_trace.log", error_type)
        return

    print("─" * 60)

    if response_text:
        logger.info("🤖 Agent response: %s", response_text.strip())
        print(f"\n🤖 Agent Response: {response_text}")
    else:
        logger.warning("⚠️  No response text received from agent.")

    # ── Step 4: Plugin Summary ────────────────
    print("\n" + "#"*60)
    print("📊 PLUGIN SUMMARY")
    print("#"*60)
    logger.info("📊 Total tool calls: %d", plugin.count)
    print(
        f"  Total tool calls recorded : {plugin.count}\n"
        + (
            "  ✅ Captured by ToolCounterPlugin!"
            if plugin.count > 0
            else "  ⚠️  No tool calls recorded — check instruction/query alignment."
        )
    )
    print("\n✅ Dual observability demo complete.")
    logger.info("✅ Run complete. Full log → agent_prod.log")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
