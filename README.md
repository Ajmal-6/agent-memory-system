Agent Memory & Forgetting System
================================

Project Description
-------------------
This project implements an autonomous AI agent that intelligently manages its own memory lifecycle.
The agent can store, retrieve, evaluate, and forget information automatically based on usefulness
and relevance, without any manual intervention.

The system features a dual-memory architecture with automated forgetting mechanisms that decide
what information to retain and what to discard based on multiple scoring factors.

Features
--------
1. Short-Term Memory.

2. Long-Term Memory.

3. Memory Scoring System

4. Automated Forgetting Mechanism

5. Memory Lifecycle Logging

Technologies Used
-----------------
- Python 3.10+
- ChromaDB (vector database for semantic search)
- Sentence Transformers (all-MiniLM-L6-v2 for embeddings)
- Transformers (optional LLM integration)
- PyYAML (configuration management)
- JSON (structured logging and data exchange)
- NumPy (numerical computations)


How to Run
----------
Step 1: Create Virtual Environment
  Windows:
    python -m venv venv
    venv\Scripts\activate

Step 2: Install Dependencies
  pip install -r requirements.txt

Step 3: Run the System

  Interactive Mode 
    python main.py


Step 4: Interact with the System
  - Type queries to chat with the agent
  - Use commands:
    * 'stats' - View memory statistics
    * 'logs' - Export memory logs
    * 'task' - Run a complete task
    * 'help' - Show available commands
    * 'quit' - Exit the system
