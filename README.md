# Kaggle AI Agents Intensive (Google)

Welcome to my repository for the **5-Day AI Agents Intensive Course with Google** by Kaggle. 

While the course provides web-based Jupyter Notebooks on Kaggle, this repository tracks my journey rebuilding, engineering, and expanding these agentic workflows **locally** inside an IDE (VS Code). This approach prioritizes production-ready software design patterns, true asynchronous logic orchestration, clean separation of concerns, and modular debugging.

---

## 🛠️ Key Architectural Highlights
* **Production-Ready Setup:** Migrated from transient Kaggle Notebook containers to a local, version-controlled Python virtual environment (`venv`).
* **Asynchronous Orchestration:** Handled the asynchronous runtime environment required by the Google Agent Development Kit (ADK) using native Python `asyncio`.
* **Security-First Architecture:** Implemented localized secret handling using `python-dotenv` to safeguard Google AI Studio credentials, ensuring zero exposure to public version control.
* **Modular Engineering:** Transitioned monolithic notebook files into discrete, maintainable components (separate files for agents, tools, and execution run loops).
* **Environment Strategy:** Each agent is implemented in two variants to bridge local development and production-ready cloud deployment:
    * `_local.py`: Optimized for **Ollama (llama3.2)**. Perfect for offline development and privacy-centric workflows.
    * `_gemini.py`: Optimized for **Google Gemini (Cloud API)**, utilizing built-in Google code execution and advanced cloud reasoning.

---

🚀 Learning Progression
Day 1: Foundations - Establishing the agent lifecycle, prompt engineering, and multi-agent routing.

Day 2: Agent Tools - Bridging AI to the digital ecosystem through Function Calling, MCP, and human-in-the-loop workflows.

Day 3: Context Engineering - Enabling persistent intelligence through session management, memory compaction, and long-term storage.

## 📂 Repository Structure

```text
kaggle-ai-agents/
│
├── .gitignore                # Prevents tracking of venv, secrets, and local databases
├── .env                      # Local environment secrets (Git ignored)
├── README.md                 # Project documentation
│
├── day_01_foundations/
│   ├── 00_test.py            # Initial API connection validation
│   ├── 01_single_agent.py    # Autonomous Search Agent
│   ├── 02_multi_agent.py     # Hierarchical Multi-Agent routing
│   ├── 03_sequential_agent.py# Sequential execution flow
│   ├── 04_parallel_agent.py  # Concurrent agent execution
│   └── 05_loop_agent.py      # Iterative output refinement
│
├── day_02_agent_tools/
│   ├── 01_custom_tools_...   # Custom local Python function integration
│   ├── 02_calculation_agent_...# Bypassing LLM math hallucinations
│   ├── 03_mcp_agent_...      # Model Context Protocol (MCP) integration
│   ├── 04_lro_agent_...      # Long-Running Operations (LRO) architecture
│   └── 05_lro_agent_full_... # Complete LRO execution loop with approvals
│
├── day_03_context_engineering/
│   ├── 01_sessions_...       # State persistence across agent interactions
│   ├── 02_compaction_...     # Optimizing context window via history compaction
│   ├── 03_session_state_...  # Advanced state serialization
│   ├── 04_memory_...         # Implementing long-term semantic memory
│   ├── 05_memory_retrieval_...# Vector-based memory retrieval systems
│   └── 06_auto_memory_...    # Autonomous memory management workflows
│
├── day_04_evaluation/        # (Upcoming) Testing and iterating agent behaviors
└── day_05_capstone/          # (Upcoming) Final production-grade deployment