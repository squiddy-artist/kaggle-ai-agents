# Kaggle AI Agents Intensive (Google)

Welcome to my repository for the **5-Day AI Agents Intensive Course with Google** by Kaggle. 

While the course provides web-based Jupyter Notebooks on Kaggle, this repository tracks my journey rebuilding, engineering, and expanding these agentic workflows **locally** inside an IDE (VS Code). This approach prioritizes production-ready software design patterns, true asynchronous logic orchestration, clean separation of concerns, and modular debugging.

---

## 🛠️ Key Architectural Highlights
* **Production-Ready Setup:** Migrated from transient Kaggle Notebook containers to a local, version-controlled Python virtual environment (`venv`).
* **Asynchronous Orchestration:** Handled the asynchronous runtime environment required by the Google Agent Development Kit (ADK) using native Python `asyncio`.
* **Security-First Architecture:** Implemented localized secret handling using `python-dotenv` to safeguard Google AI Studio credentials, ensuring zero exposure to public version control.
* **Modular Engineering:** Transitioned monolithic notebook files into discrete, maintainable components (separate files for agents, tools, and execution run loops).

---

## 📂 Repository Structure

The project is chronologically organized to map directly to the course timeline and learning progression:

```text
kaggle-ai-agents/
│
├── .gitignore                 # Prevents tracking of virtual environments and secrets
├── .env                       # Local environment secrets (API Keys - Git ignored)
├── README.md                  # Project documentation
│
├── day_01_foundations/
│   ├── 00_test.py             # Initial API connection and environment validation
│   ├── 01_single_agent.py     # Autonomous Search Agent utilizing the ReAct framework
│   ├── 02_multi_agent.py      # Hierarchical Multi-Agent routing (Manager/Worker pattern)
│   ├── 03_sequential_agent.py # Sequential execution flow passing state between specialized agents
│   ├── 04_parallel_agent.py   # Concurrent agent execution mapping for speed optimization
│   └── 05_loop_agent.py       # Loop-based agent evaluation and iterative output refinement
│
├── day_02_agent_tools/
│   ├── 01_custom_tools.py     # Agent integration with custom local Python functions
│   ├── 02_calculation_agent.py# Bypassing LLM math hallucinations via BuiltInCodeExecutor
│   ├── 03_mcp_agent.py        # Model Context Protocol (MCP) integration via Node.js
│   ├── 04_lro_agent.py        # Architecture setup for Long-Running Operations (LRO)
│   └── 05_lro_agent_full.py   # Complete LRO execution loop with human-in-the-loop approvals
│
├── day_03_frameworks/         # (Upcoming) Advanced Agentic Orchestration Frameworks
├── day_04_evaluation/         # (Upcoming) Testing and iterating agent behaviors
└── day_05_capstone/           # (Upcoming) The final production-grade Agent deployment