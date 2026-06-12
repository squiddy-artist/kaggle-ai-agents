import ast
import asyncio
import logging
import os
import sys
import traceback
import warnings
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
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
logger = logging.getLogger("agent_local")
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
# Append mode — survives multiple runs, audit trail intact
_file = logging.FileHandler("agent_local.log", mode="a", encoding="utf-8")
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
warnings.filterwarnings(
    "ignore",
    message=".*EXPERIMENTAL.*",
    category=UserWarning,
    module="google.*"
)


# ─────────────────────────────────────────────
# 📌 Constants
# ─────────────────────────────────────────────
APP_NAME   = "local_prod_agent"
USER_ID    = "nandakumar_dev"
SESSION_ID = "observability-session-01"


# ─────────────────────────────────────────────
# 🔌 Custom Plugin — ToolCounterPlugin
# **kwargs pattern — version-resilient
# Key names confirmed from DEBUG run:
#   'tool', 'tool_args', 'tool_context', 'result'
# ─────────────────────────────────────────────
class ToolCounterPlugin(BasePlugin):
    """
    Custom production monitoring plugin.
    Tracks every tool call made by the agent in real time.
    """

    def __init__(self):
        super().__init__(name="tool_counter")
        self.count = 0

    async def after_tool_callback(self, **kwargs):
        """
        Fires automatically after every tool call.
        Uses logger instead of print() — respects LOG_LEVEL + writes to file.
        Key names confirmed from live DEBUG output:
          - 'tool_args'  (not 'args')
          - 'result'     (not 'tool_response')
        """
        self.count += 1

        tool      = kwargs.get('tool')
        tool_name = getattr(tool, 'name', 'unknown')
        args      = kwargs.get('tool_args', {})   # confirmed: not 'args'
        response  = kwargs.get('result', 'N/A')   # confirmed: not 'tool_response'

        # Terminal: one clean INFO line per tool call
        logger.info(
            "🔧 Tool fired: '%s' | args: %s | result: %s | total_calls: %d",
            tool_name, args, response, self.count
        )
        # File only: full kwargs keys for future debugging
        logger.debug("Tool kwargs keys: %s", list(kwargs.keys()))


# ─────────────────────────────────────────────
# 🛠️ Tool Definition
# Full docstring is CRITICAL for local models —
# llama3.2 needs explicit Args/Returns to reliably
# call tools vs just answering in plain text
# ─────────────────────────────────────────────
def count_papers(papers: List[str]) -> int:
    """
    Counts the number of research papers in a given list.

    Args:
        papers: A list of paper titles or identifiers to count.
                Also accepts a string representation of a list
                e.g. "['Paper A', 'Paper B']" — common from local models.

    Returns:
        The total number of papers as an integer.
    """
    # Local model safety net — llama3.2 often passes lists as strings
    if isinstance(papers, str):
        try:
            papers = ast.literal_eval(papers)
            logger.debug("📄 string input parsed → list via ast.literal_eval")
        except (ValueError, SyntaxError):
            papers = [papers]
            logger.debug("📄 unparseable string → wrapped as single-item list")

    result = len(papers) if isinstance(papers, list) else 0
    logger.info("📄 count_papers → %d items → result: %d", len(papers), result)
    logger.debug("Papers received: %s", papers)
    return result


# ─────────────────────────────────────────────
# 🤖 Agent Definition
# Explicit instruction — local models need more
# hand-holding than cloud models to use tools
# ─────────────────────────────────────────────
research_agent = Agent(
    name="research_agent",
    model=LiteLlm(model="ollama_chat/llama3.2"),
    instruction=(
        "You are a research assistant. When the user gives you a list of papers, "
        "you MUST call the count_papers tool. "
        "Pass the papers as a proper JSON array of strings. "
        "Example: count_papers(papers=['Paper A', 'Paper B', 'Paper C']). "
        "IMPORTANT: papers must be a LIST, not a string. "
        "Do NOT pass papers='[...]' as a single string value. "
        "Then report the exact number returned by the tool."
    ),
    tools=[FunctionTool(func=count_papers)]
)

# ─────────────────────────────────────────────
# 🚀 Main Async Runner
# ─────────────────────────────────────────────
async def main():

    print("\n" + "#"*60)
    print("🔌 DEMO: Dual Plugin Observability (Local Ollama)")
    print(f"   Model     : ollama_chat/llama3.2")
    print(f"   Log Level : {LOG_LEVEL}")
    print(f"   Log File  : agent_local.log  ← full DEBUG trace always saved")
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

    # ── Step 2: Runner — plugins controlled by LOG_LEVEL ──────────
    # LoggingPlugin uses print() internally — bypasses logging system
    # Only attach in DEBUG mode where verbosity is expected
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
    logger.info("📌 Running local Ollama agent...")
    print("─" * 60)

    query = types.Content(
        role="user",
        parts=[types.Part(
            text=(
                "Count these papers: ['Paper A', 'Paper B', 'Paper C']. "
                "Use the count_papers tool with the full list."
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
        # Not an error — user stopped it deliberately
        logger.warning("⚠️  Run interrupted by user.")
        return

    except Exception as e:
        error_type = type(e).__name__
        error_msg  = str(e)

        # ── Ollama not running: most common local failure ──────────
        if any(k in error_msg for k in ["Connection refused", "ConnectError", "ollama"]):
            print(f"\n{'─'*60}")
            print(f"🦙  OLLAMA NOT REACHABLE")
            print(f"    Fix 1  : Start Ollama   →  ollama serve")
            print(f"    Fix 2  : Pull the model →  ollama pull llama3.2")
            print(f"    Fix 3  : Verify running →  ollama list")
            print(f"{'─'*60}")
            logger.warning("Ollama unreachable — full error saved to error_trace.log")
            with open("error_trace.log", "w", encoding="utf-8") as f:
                f.write("OLLAMA CONNECTION ERROR\n\n")
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
    logger.info("✅ Run complete. Full log → agent_local.log")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
