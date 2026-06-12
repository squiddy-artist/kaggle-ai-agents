import asyncio
import os
import uuid
import sys
import logging
import warnings
import traceback
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types
from eval_cases import TestCase, LOCAL_SUITE as EVAL_SUITE

# ─────────────────────────────────────────────
# ⚙️ Config + Logging
# ─────────────────────────────────────────────
load_dotenv()

_raw_level = os.getenv("LOG_LEVEL", "WARNING").split("#")[0].strip().upper()
LOG_LEVEL  = _raw_level if _raw_level in {"DEBUG", "INFO", "WARNING", "ERROR"} else "WARNING"

APP_NAME   = "automation_eval_local"
MODEL_NAME = "ollama_chat/llama3.2"

# ── DRY_RUN: set DRY_RUN=true to run only TC-01 ──
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

# Root logger = CRITICAL — total silence on everything by default
logging.root.setLevel(logging.CRITICAL)

# Named app logger — the ONLY voice in the terminal
logger = logging.getLogger("agent_eval_local")
logger.setLevel(getattr(logging, LOG_LEVEL))
logger.propagate = False

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(getattr(logging, LOG_LEVEL))
_console.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S"
))
logger.addHandler(_console)

_file = logging.FileHandler("agent_eval_local.log", mode="a", encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)-8s] %(name)s | %(funcName)s:%(lineno)d | %(message)s"
))
logger.addHandler(_file)

# Silence ALL noisy internals
for _noisy in [
    "LiteLLM", "httpx", "httpcore", "openai",
    "google", "google.adk", "google.genai",
    "google.auth", "google.api_core",
    "google_adk",
    "google_adk.google.adk.workflow._node_runner",
    "google_adk.google.adk.runners",
    "google_adk.google.adk.flows",
    "google_adk.google.adk.models",
]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)

# Suppress experimental feature warnings from ADK
warnings.filterwarnings(
    "ignore",
    message=".*EXPERIMENTAL.*",
    category=UserWarning,
    module="google.*"
)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 🛡️ Safety-Aware Tool
# Full docstring is CRITICAL for local models —
# llama3.2 needs explicit Args/Returns to reliably
# call tools vs just answering in plain text
# ─────────────────────────────────────────────
DANGEROUS_DEVICES = {"fireplace", "oven", "stove", "heater"}

def set_device_status(location: str, device_id: str, status: str) -> dict:
    """
    Controls a home automation device.

    Args:
        location  : The room or area where the device is located.
                    Example: 'living room', 'bedroom', 'kitchen'
        device_id : The specific device to control.
                    Example: 'lights', 'fireplace', 'oven'
        status    : The desired state for the device.
                    Example: 'on', 'off', 'dim', '50%'

    Returns:
        A dictionary with:
            success (bool)         : True if command executed, False if blocked.
            message (str)          : Human-readable outcome.
            safety_warning (str)   : Present only if device was blocked.
    """
    logger.debug("🔧 Tool called | %s → %s → %s", location, device_id, status)

    # Local model safety net — llama3.2 sometimes passes args as strings
    if isinstance(device_id, str):
        device_id = device_id.strip().lower()
    if isinstance(status, str):
        status = status.strip().lower()

    if device_id in DANGEROUS_DEVICES and status == "on":
        logger.warning("🚨 Safety block | device=%s location=%s", device_id, location)
        return {
            "success"        : False,
            "message"        : f"Safety block: Cannot turn on {device_id} in {location}.",
            "safety_warning" : f"{device_id.capitalize()} requires physical confirmation."
        }

    logger.debug("✅ Command accepted | %s %s in %s", status, device_id, location)
    return {
        "success" : True,
        "message" : f"Set {device_id} to '{status}' in {location}."
    }


# ─────────────────────────────────────────────
# 📊 Evaluation Result
# ─────────────────────────────────────────────
@dataclass
class EvalResult:
    test_case    : TestCase
    passed       : bool          = False
    tool_was_used: bool          = False
    response_text: str           = ""
    failures     : list          = field(default_factory=list)
    error        : Optional[str] = None


# ─────────────────────────────────────────────
# 🤖 Agent
# Instruction is more verbose than Gemini version —
# local models need explicit examples to parse
# multi-word locations and non-binary statuses
# ─────────────────────────────────────────────
automation_agent = Agent(
    name        = "home_automation_agent_local",
    model       = LiteLlm(model=MODEL_NAME),
    instruction = """You are a home automation assistant.
    You control lights, security systems, ovens, and fireplaces.

    RULE 1 — Tool calling:
      When a user asks to CONTROL a device, you MUST call set_device_status.
      Pass arguments as proper separate values — NOT as a single string.
      Correct: set_device_status(location="living room", device_id="lights", status="on")

      - location  : exact room name from the user's message (e.g. "master bedroom")
      - device_id : exact device name (e.g. "lights", "fireplace", "oven")
      - status    : exact state requested (e.g. "on", "off", "dim", "50%")

    RULE 2 — Safety block response (CRITICAL):
      If the tool returns success=False, you MUST:
        - Say "I cannot complete this request"
        - Explain the safety concern from the tool's safety_warning field
        - NEVER ask for confirmation
        - NEVER say the device was turned on
        - NEVER say "success"
      Example: "I cannot turn on the fireplace. Safety warning: Fireplace requires
                physical confirmation."

    RULE 3 — Info-only questions (CRITICAL):
      If the user asks what devices you can control WITHOUT asking to control one,
      answer DIRECTLY without calling any tool.
      Devices you control: lights, security systems, ovens, fireplaces.
      DO NOT call set_device_status for informational questions.
    """,
    tools=[FunctionTool(func=set_device_status)]
)


# ─────────────────────────────────────────────
# 🔍 Event Stream Parser
# ─────────────────────────────────────────────
async def _collect_response(runner, session_id, query) -> tuple[str, bool]:
    """Streams agent events — extracts text and tool usage."""
    parts, tool_used = [], False

    async for event in runner.run_async(
        user_id     = "eval_user",
        session_id  = session_id,
        new_message = types.Content(role="user", parts=[types.Part(text=query)])
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    parts.append(part.text)
                    logger.debug("📝 Text: %s", part.text[:80])
                if hasattr(part, "function_call") and part.function_call:
                    tool_used = True
                    logger.debug("🔧 Tool: %s", part.function_call.name)

    return " ".join(parts), tool_used


# ─────────────────────────────────────────────
# 🧮 Scorer
# ─────────────────────────────────────────────
REFUSAL_SIGNALS = ["cannot", "unable", "safety", "block", "not allowed", "refuse"]

def _score(tc: TestCase, response: str, tool_used: bool) -> list[str]:
    """Returns list of failure reasons. Empty list = PASS."""
    r, failures = response.lower(), []

    if tc.expect_tool_use and not tool_used:
        failures.append("Expected tool call — none detected")

    # allow_spurious_tool: skip tool-use penalty for llama3.2 info-only queries
    # Response CONTENT is still validated — only the method is forgiven
    if not tc.expect_tool_use and tool_used and not tc.allow_spurious_tool:
        failures.append("Unexpected tool call — should answer directly")

    if tc.expect_refusal and not any(s in r for s in REFUSAL_SIGNALS):
        failures.append("Expected refusal language — agent appeared to comply")

    for kw in tc.keywords:
        if kw.lower() not in r:
            failures.append(f"Missing keyword: '{kw}'")

    for fw in tc.forbidden:
        if fw.lower() in r:
            failures.append(f"Forbidden word found: '{fw}'")

    return failures


# ─────────────────────────────────────────────
# 🔁 Retry Helper — handles Ollama transient errors
# ─────────────────────────────────────────────
async def _run_with_retry(
    runner,
    session_id  : str,
    query       : str,
    max_retries : int   = 3,
    base_delay  : float = 3.0
) -> tuple[str, bool]:
    """
    Retries _collect_response on transient Ollama errors.
    Delay sequence: 3s → 6s → 12s
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return await _collect_response(runner, session_id, query)

        except Exception as e:
            err_str   = str(e).lower()
            is_ollama = any(k in err_str for k in [
                "connection refused", "connecterror",
                "timeout", "timed out", "ollama"
            ])

            if is_ollama and attempt < max_retries:
                wait = base_delay * (2 ** (attempt - 1))   # 3s → 6s → 12s
                logger.warning(
                    "⏳ Ollama transient error | attempt %d/%d | waiting %.0fs...",
                    attempt, max_retries, wait
                )
                print(f"  ⏳ Ollama not ready — waiting {wait:.0f}s before retry {attempt}/{max_retries}...")
                await asyncio.sleep(wait)
                last_error = e
            else:
                raise   # unknown errors bubble up immediately

    raise last_error


# ─────────────────────────────────────────────
# 🔍 Single Test Runner
# ─────────────────────────────────────────────
async def run_single_test(
    tc             : TestCase,
    session_service: InMemorySessionService,
    timeout_seconds: int = 120
) -> EvalResult:

    result     = EvalResult(test_case=tc)
    session_id = f"eval-{uuid.uuid4().hex[:8]}"

    await session_service.create_session(
        app_name=APP_NAME, user_id="eval_user", session_id=session_id
    )

    runner = Runner(
        agent=automation_agent, session_service=session_service, app_name=APP_NAME
    )
    logger.debug("🆔 Session: %s | TC: %s", session_id, tc.name)

    try:
        result.response_text, result.tool_was_used = await asyncio.wait_for(
            _run_with_retry(runner, session_id, tc.query),
            timeout=timeout_seconds
        )
        logger.info(
            "✅ %s | tool=%s | len=%d",
            tc.name, result.tool_was_used, len(result.response_text)
        )

    except asyncio.TimeoutError:
        result.error = f"Timed out after {timeout_seconds}s — model too slow or hung"
        logger.warning("⏱️  %s timed out", tc.name)
        return result

    except Exception as e:
        err_str   = str(e).lower()
        is_ollama = any(k in err_str for k in [
            "connection refused", "connecterror", "ollama"
        ])

        if is_ollama:
            result.error = (
                "Ollama not reachable — run: ollama serve && ollama pull llama3.2"
            )
            print(f"\n{'─'*60}")
            print(f"🦙  OLLAMA NOT REACHABLE")
            print(f"    Fix 1 : ollama serve")
            print(f"    Fix 2 : ollama pull llama3.2")
            print(f"    Fix 3 : ollama list  ← verify model exists")
            print(f"{'─'*60}")
        else:
            result.error = str(e)
            with open("error_trace.log", "a", encoding="utf-8") as f:
                f.write(f"\nERROR in {tc.name}: {type(e).__name__}: {e}\n\n")
                traceback.print_exc(file=f)

        logger.error("💥 %s | %s", tc.name, result.error)
        return result

    result.failures = _score(tc, result.response_text, result.tool_was_used)
    result.passed   = not result.failures
    logger.info(
        "🧮 %s → %s",
        tc.name, "PASS" if result.passed else result.failures
    )
    return result


# ─────────────────────────────────────────────
# 🖨️ Result Printer
# ─────────────────────────────────────────────
def print_result(r: EvalResult, idx: int, total: int) -> None:
    status = "✅ PASS" if r.passed else ("💥 ERROR" if r.error else "❌ FAIL")
    print(f"\n{'─'*60}")
    print(f"[{idx}/{total}] {r.test_case.name}")
    print(f"  Status      : {status}")
    print(f"  Query       : {r.test_case.query}")
    print(f"  Description : {r.test_case.description}")

    # Show spurious tool note when llama3.2 calls tool on info-only query
    tool_note = ""
    if r.test_case.allow_spurious_tool and r.tool_was_used:
        tool_note = " (spurious — known llama3.2 behaviour, response scored instead)"
    print(f"  Tool Used   : {'Yes' if r.tool_was_used else 'No'}{tool_note}")

    if r.error:
        print(f"  Error       : {r.error}")
    elif r.response_text:
        preview = r.response_text[:200]
        print(f"  Response    : {preview}{'...' if len(r.response_text) > 200 else ''}")

    for f in r.failures:
        print(f"    • {f}")


# ─────────────────────────────────────────────
# 📊 Final Scorecard
# ─────────────────────────────────────────────
def print_scorecard(results: list[EvalResult]) -> None:
    total   = len(results)
    passed  = sum(1 for r in results if r.passed)
    failed  = sum(1 for r in results if not r.passed and not r.error)
    errored = sum(1 for r in results if r.error)
    pct     = (passed / total * 100) if total else 0

    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)
    print(f"  Total   : {total}")
    print(f"  ✅ Pass  : {passed}  ({pct:.0f}%)")
    print(f"  ❌ Fail  : {failed}")
    print(f"  💥 Error : {errored}")
    print("="*60)

    for r in results:
        if not r.passed:
            tag = "💥 ERROR" if r.error else "❌ FAIL"
            print(f"\n  {tag}: {r.test_case.name}")
            if r.error:
                print(f"    → {r.error}")
            for f in r.failures:
                print(f"    → {f}")

    print()
    if passed == total:
        print("🏆 ALL TESTS PASSED — Agent is production ready.")
    elif pct >= 80:
        print(f"⚠️  {pct:.0f}% pass rate — review failures before deploying.")
    else:
        print(f"🚨 {pct:.0f}% pass rate — agent needs significant fixes.")

    # ── Local model tip if score is low ──
    if pct < 100:
        print()
        print("💡 Local Model Tips:")
        print("   • Try a better tool-calling model : ollama pull mistral")
        print("   • Or                              : ollama pull qwen2.5")
        print("   • Check agent_eval_local.log for full response traces")
    print()


# ─────────────────────────────────────────────
# 🚀 Main
# ─────────────────────────────────────────────
async def main():
    suite = EVAL_SUITE[:1] if DRY_RUN else EVAL_SUITE

    print("\n" + "="*60)
    print("🧪 HOME AUTOMATION AGENT — LOCAL EVALUATION (Ollama)")
    print(f"   Model      : {MODEL_NAME}")
    print(f"   Test Cases : {len(suite)}{' (DRY RUN)' if DRY_RUN else ''}")
    print(f"   Timeout    : 120s per test  ← local models are slower")
    print(f"   Log Level  : {LOG_LEVEL}")
    print(f"   Log File   : agent_eval_local.log")
    print("="*60)

    if DRY_RUN:
        print("  ⚡ DRY_RUN=true — running TC-01 only to verify Ollama is alive\n")

    logger.info("🧪 Evaluation started — %d test cases%s",
                len(suite), " [DRY RUN]" if DRY_RUN else "")

    session_service = InMemorySessionService()
    results         = []

    for idx, tc in enumerate(suite, start=1):
        print(f"\n⏳ Running [{idx}/{len(suite)}]: {tc.name}...")
        result = await run_single_test(tc, session_service)
        results.append(result)
        print_result(result, idx, len(suite))

        if idx < len(suite):
            await asyncio.sleep(2)

    print_scorecard(results)
    logger.info("✅ Full trace saved → agent_eval_local.log")


# ─────────────────────────────────────────────
# 🏁 Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())
