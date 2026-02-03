#!/usr/bin/env python3
"""
Main entry point for Agent Memory System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config_loader import load_config
from src.memory.manager import MemoryManager
from src.agent.base_agent import BaseAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main function to run the agent memory system."""

    print("=" * 60)
    print("Agent Memory & Forgetting System")
    print("=" * 60)

    # Load configuration
    try:
        config = load_config()
        print(f"✓ Configuration loaded from config.yaml")
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        print("Using default configuration...")
        config = {
            'memory': {
                'short_term': {'capacity': 10, 'ttl_seconds': 3600},
                'long_term': {
                    'vector_db': 'chromadb',
                    'collection_name': 'agent_memory',
                    'persist_directory': None
                },
                'scoring': {
                    'weights': {
                        'recency': 0.3,
                        'frequency': 0.2,
                        'relevance': 0.4,
                        'task_success': 0.1
                    },
                    'recency_decay_rate': 0.1,
                    'frequency_window': 100
                },
                'forgetting': {
                    'short_term_threshold': 0.2,
                    'long_term_threshold': 0.3,
                    'cleanup_interval': 50,
                    'batch_size': 10
                }
            }
        }

    # Initialize memory manager
    try:
        memory_manager = MemoryManager(config)
        print(f"✓ Memory manager initialized")
    except Exception as e:
        print(f"✗ Error initializing memory manager: {e}")
        return

    # Initialize agent
    try:
        agent = BaseAgent(config, memory_manager)
        print(f"✓ Agent initialized")
    except Exception as e:
        print(f"✗ Error initializing agent: {e}")
        return

    # Display initial stats
    try:
        stats = memory_manager.get_stats()
        print(f"\nInitial Memory Stats:")
        print(f"  Short-term capacity: {len(memory_manager.short_term_memory)}/{config['memory']['short_term']['capacity']}")
        print(f"  Long-term memories: {stats['current_long_term']}")
        print(f"  Forgetting thresholds: Short={config['memory']['forgetting']['short_term_threshold']}, "
              f"Long={config['memory']['forgetting']['long_term_threshold']}")
    except Exception as e:
        print(f"✗ Error getting stats: {e}")

    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("Commands: 'quit' to exit, 'stats' for statistics, 'logs' to export logs, 'task' to run a task")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'stats':
                try:
                    stats = agent.get_agent_stats()
                    print(f"\nAgent Statistics:")
                    print(f"  Conversations: {stats['conversations']}")
                    print(f"  Tasks completed: {stats['tasks_completed']}")
                    print(f"  Total memories created: {stats['memory_stats']['total_created']}")
                    print(f"  Total memories forgotten: {stats['memory_stats']['total_forgotten']}")
                    print(f"  Current short-term: {stats['memory_stats']['current_short_term']}")
                    print(f"  Current long-term: {stats['memory_stats']['current_long_term']}")
                except Exception as e:
                    print(f"Error getting stats: {e}")
                continue
            elif user_input.lower() == 'logs':
                try:
                    log_file = memory_manager.export_logs()
                    print(f"✓ Logs exported to {log_file}")
                except Exception as e:
                    print(f"Error exporting logs: {e}")
                continue
            elif user_input.lower() == 'task':
                try:
                    task_desc = input("Enter task description: ")
                    result = agent.run_task(task_desc)
                    print(f"\nTask completed: {result['task']}")
                    print(f"Steps: {len(result['steps'])}")
                except Exception as e:
                    print(f"Error running task: {e}")
                continue
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  help - Show this help")
                print("  stats - Show agent statistics")
                print("  logs - Export memory logs")
                print("  task - Run a task")
                print("  quit - Exit the program")
                print("  <any text> - Chat with the agent")
                continue

            # Process query
            try:
                response = agent.process_query(user_input)
                print(f"\nAgent: {response}")
            except Exception as e:
                print(f"Error processing query: {e}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    main()