#!/usr/bin/env python3
"""
Demonstration script for the Agent Memory System
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import load_config
from src.memory.manager import MemoryManager
from src.agent.base_agent import BaseAgent

def run_demo():
    """Run a complete demonstration of the system."""
    
    print("🚀 Starting Agent Memory System Demo")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize system
    memory_manager = MemoryManager(config)
    agent = BaseAgent(config, memory_manager)
    
    # Demo queries
    demo_queries = [
        ("Hello, who are you?", "introduction"),
        ("I want to remember that Python is a programming language", "learning"),
        ("What did I tell you about Python?", "recall"),
        ("Remember that AI agents need memory to be effective", "important"),
        ("How does your memory system work?", "explanation"),
        ("What programming language did I mention earlier?", "recall"),
        ("Forget everything about Python", "modification"),
        ("What do you know about Python now?", "verification"),
        ("Remember that machine learning is a subset of AI", "learning"),
        ("What's the weather like?", "general"),
        ("Tell me about AI", "general"),
        ("What have we discussed about AI?", "recall"),
    ]
    
    print("\n📝 Running demonstration queries:")
    print("-" * 40)
    
    for i, (query, context) in enumerate(demo_queries, 1):
        print(f"\n{i}. Query: {query}")
        print(f"   Context: {context}")
        
        response = agent.process_query(query, task_context=context)
        print(f"   Response: {response}")
        
        # Show memory stats every few queries
        if i % 3 == 0:
            stats = memory_manager.get_stats()
            print(f"\n   📊 Memory Stats: {stats['current_short_term']} short-term, "
                  f"{stats['current_long_term']} long-term, "
                  f"{stats['total_forgotten']} forgotten")
        
        time.sleep(0.5)  # Small delay for readability
    
    # Run a complete task
    print("\n" + "=" * 60)
    print("🎯 Running a complete task example:")
    print("-" * 40)
    
    task_result = agent.run_task("Explain how memory systems work in AI agents")
    print(f"Task: {task_result['task']}")
    print(f"Completed in {len(task_result['steps'])} steps")
    print(f"Success: {task_result['success']}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("📈 Final System Statistics:")
    print("-" * 40)
    
    final_stats = agent.get_agent_stats()
    print(f"Total conversations: {final_stats['conversations']}")
    print(f"Tasks completed: {final_stats['tasks_completed']}")
    print(f"Memories created: {final_stats['memory_stats']['total_created']}")
    print(f"Memories forgotten: {final_stats['memory_stats']['total_forgotten']}")
    print(f"Current short-term memories: {final_stats['memory_stats']['current_short_term']}")
    print(f"Current long-term memories: {final_stats['memory_stats']['current_long_term']}")
    
    # Export logs
    log_file = memory_manager.export_logs("./logs/demo_export.json")
    print(f"\n📁 Logs exported to: {log_file}")
    
    print("\n✅ Demo completed successfully!")
    print("Check the 'logs' directory for detailed memory events.")

if __name__ == "__main__":
    run_demo()