# Kaggle AI Agents Intensive (Google)

Welcome to my repository for the **5-Day AI Agents Intensive Course with Google** by Kaggle.

While the course provides web-based Jupyter Notebooks on Kaggle, this repository tracks
my journey rebuilding, engineering, and expanding these agentic workflows **locally**
inside an IDE (VS Code). This approach prioritizes production-ready software design
patterns, true asynchronous logic orchestration, clean separation of concerns, and
modular debugging.

---

## 🛠️ Key Architectural Highlights

- **Production-Ready Setup:** Migrated from transient Kaggle Notebook containers to a
  local, version-controlled Python virtual environment (`venv`).
- **Asynchronous Orchestration:** Handled the asynchronous runtime environment required
  by the Google Agent Development Kit (ADK) using native Python `asyncio`.
- **Security-First Architecture:** Implemented localized secret handling using
  `python-dotenv` to safeguard Google AI Studio credentials, ensuring zero exposure to
  public version control.
- **Modular Engineering:** Transitioned monolithic notebook files into discrete,
  maintainable components — separate files for agents, tools, evaluation suites, and
  execution runners.
- **Dual-Runner Strategy:** Every agent is implemented in two variants to bridge local
  development and cloud deployment:
  - `_local.py` — Optimized for **Ollama (llama3.2)**. Perfect for offline development
    and privacy-centric workflows.
  - `_gemini.py` — Optimized for **Google Gemini (Cloud API)**, utilizing built-in
    Google code execution and advanced cloud reasoning.
- **Systematic Evaluation:** Built a structured agent evaluation framework with a shared
  test suite, 4-point scorer, safety gate validation, and dual-runner support — achieving
  100% pass rate on llama3.2 locally.

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/your-username/kaggle-ai-agents.git
cd kaggle-ai-agents
pip install -e .
```

### 2. Configure Secrets

```bash
cp .env.example .env
# Open .env and add your GOOGLE_API_KEY
```

### 3. Run Any Agent

```bash
# Local (Ollama must be running)
python day_04_agent_quality/03_evaluation_local.py

# Cloud (Gemini API key required)
python day_04_agent_quality/03_evaluation_gemini.py
```

---

## 📂 Repository Structure

```text
kaggle-ai-agents/
│
├── pyproject.toml                 # Dependency manifest (pip install -e .)
├── .env.example                   # Safe secrets template (copy → .env)
├── .env                           # Local secrets — Git ignored
├── .gitignore                     # Excludes venv, secrets, logs, cache
├── README.md                      # Project documentation
│
├── day_01_foundations/
│   ├── 00_test.py                 # Initial API connection validation
│   ├── 01_single_agent.py         # Autonomous Search Agent
│   ├── 02_multi_agent.py          # Hierarchical Multi-Agent routing
│   ├── 03_sequential_agent.py     # Sequential execution flow
│   ├── 04_parallel_agent.py       # Concurrent agent execution
│   └── 05_loop_agent.py           # Iterative output refinement
│
├── day_02_agent_tools/
│   ├── 01_custom_tools_...        # Custom local Python function integration
│   ├── 02_calculation_agent_...   # Bypassing LLM math hallucinations
│   ├── 03_mcp_agent_...           # Model Context Protocol (MCP) integration
│   ├── 04_lro_agent_...           # Long-Running Operations (LRO) architecture
│   └── 05_lro_agent_full_...      # Complete LRO execution loop with approvals
│
├── day_03_context_engineering/
│   ├── 01_sessions_...            # State persistence across agent interactions
│   ├── 02_compaction_...          # Context window optimisation via history compaction
│   ├── 03_session_state_...       # Advanced state serialisation
│   ├── 04_memory_...              # Long-term semantic memory implementation
│   ├── 05_memory_retrieval_...    # Vector-based memory retrieval systems
│   └── 06_auto_memory_...         # Autonomous memory management workflows
│
├── day_04_agent_quality/
│   ├── eval_cases.py              # Shared test suite (GEMINI_SUITE + LOCAL_SUITE)
│   ├── 01_observability_gemini.py # Structured logging + OpenTelemetry (Gemini)
│   ├── 01_observability_local.py  # Structured logging + OpenTelemetry (Ollama)
│   ├── 02_production_observability_gemini.py  # Production-grade tracing (Gemini)
│   ├── 02_production_observability_local.py   # Production-grade tracing (Ollama)
│   ├── 03_evaluation_gemini.py    # Systematic eval runner — Gemini API
│   └── 03_evaluation_local.py     # Systematic eval runner — Ollama llama3.2
│
└── day_05_capstone/               # (Upcoming) Final production-grade deployment
```

---

## 📊 Evaluation Results (Day 04)

| Runner | Model | Pass Rate | Notes |
|---|---|---|---|
| Local | llama3.2 (Ollama) | 6/6 — 100% ✅ | TC-03 spurious tool call handled gracefully |
| Gemini | gemini-2.0-flash-lite | TBD | Run `03_evaluation_gemini.py` |

---

## 📅 Course Progress

| Day | Topic | Status |
|---|---|---|
| Day 01 | Foundations — Agent types & routing | ✅ Complete |
| Day 02 | Agent Tools — MCP, LRO, custom tools | ✅ Complete |
| Day 03 | Context Engineering — Memory & sessions | ✅ Complete |
| Day 04 | Agent Quality — Observability & evaluation | ✅ Complete |
| Day 05 | Capstone — Production deployment | 🔜 Upcoming |
