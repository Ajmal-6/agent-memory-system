import pytest
import tempfile
import os
from src.memory.manager import MemoryManager


class TestMemoryManager:
    @pytest.fixture
    def config(self):
        return {
            'memory': {
                'short_term': {'capacity': 5, 'ttl_seconds': 3600},
                'long_term': {
                    'vector_db': 'chromadb',
                    'collection_name': 'test_memory',
                    'persist_directory': None  # Use in-memory for tests
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
                    'cleanup_interval': 5,
                    'batch_size': 2
                }
            }
        }

    @pytest.fixture
    def memory_manager(self, config):
        # Patch ChromaDB to use mock for tests
        manager = MemoryManager(config)
        # Force mock mode for tests
        manager.collection = None
        manager.long_term_storage = {}
        manager.mock_embeddings = {}
        return manager

    def test_add_short_term_memory(self, memory_manager):
        """Test adding short-term memory."""
        memory_id = memory_manager.add_memory(
            "Test short-term memory",
            memory_type="short"
        )

        assert memory_id is not None
        assert len(memory_manager.short_term_memory) == 1
        assert memory_manager.memory_stats['current_short_term'] == 1

    def test_add_long_term_memory(self, memory_manager):
        """Test adding long-term memory."""
        memory_id = memory_manager.add_memory(
            "Test long-term memory",
            memory_type="long"
        )

        assert memory_id is not None
        stats = memory_manager.get_stats()
        assert stats['current_long_term'] == 1
        assert memory_id in memory_manager.long_term_storage

    def test_retrieve_memories(self, memory_manager):
        """Test memory retrieval."""
        # Add some memories
        mem1_id = memory_manager.add_memory("Python is a programming language", "long")
        memory_manager.add_memory("AI needs memory systems", "long")

        # Retrieve
        results = memory_manager.retrieve_memories("programming language", top_k=2)

        assert len(results) > 0
        # Should contain Python memory
        python_memory_found = any(
            "Python" in result.get('content', '') for result in results
        )
        assert python_memory_found

    def test_forget_memory(self, memory_manager):
        """Test forgetting a specific memory."""
        memory_id = memory_manager.add_memory("Memory to forget", "short")

        initial_count = len(memory_manager.short_term_memory)
        result = memory_manager.forget_memory(memory_id)

        assert result is True
        assert len(memory_manager.short_term_memory) == initial_count - 1
        assert memory_manager.memory_stats['total_forgotten'] == 1

    def test_cleanup_memories(self, memory_manager):
        """Test automatic cleanup."""
        # Add multiple memories
        for i in range(10):
            memory_manager.add_memory(f"Test memory {i}", "short")

        # Force cleanup
        forgotten = memory_manager._cleanup_memories()

        # Should forget some memories based on thresholds
        assert forgotten >= 0
        assert len(memory_manager.short_term_memory) <= 10

    def test_get_stats(self, memory_manager):
        """Test statistics retrieval."""
        # Add some memories
        memory_manager.add_memory("Test 1", "short")
        memory_manager.add_memory("Test 2", "long")

        stats = memory_manager.get_stats()

        assert 'total_created' in stats
        assert 'current_short_term' in stats
        assert 'current_long_term' in stats
        assert stats['total_created'] == 2
        assert stats['current_short_term'] == 1
        assert stats['current_long_term'] == 1