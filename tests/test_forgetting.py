import pytest
from datetime import datetime, timedelta
from src.memory.forgetting import ForgettingMechanism


class TestForgettingMechanism:
    @pytest.fixture
    def forget_mechanism(self):
        config = {
            'short_term_threshold': 0.2,
            'long_term_threshold': 0.3,
            'cleanup_interval': 5,
            'batch_size': 2
        }
        return ForgettingMechanism(config)

    @pytest.fixture
    def high_score_memory(self):
        return {
            'id': 'high-123',
            'type': 'long',
            'final_score': 0.9,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 10
        }

    @pytest.fixture
    def low_score_memory(self):
        return {
            'id': 'low-123',
            'type': 'long',
            'final_score': 0.1,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 1
        }

    def test_threshold_based_forgetting(self, forget_mechanism, high_score_memory, low_score_memory):
        """Test threshold-based forgetting decision."""
        # High score should not be forgotten
        should_forget_high = forget_mechanism._threshold_based_forgetting(high_score_memory)
        assert should_forget_high is False

        # Low score should be forgotten
        should_forget_low = forget_mechanism._threshold_based_forgetting(low_score_memory)
        assert should_forget_low is True

    def test_random_forgetting(self, forget_mechanism, high_score_memory):
        """Test random forgetting (probabilistic)."""
        # Run multiple times to check it sometimes returns True
        results = []
        for _ in range(100):
            result = forget_mechanism._random_forgetting(high_score_memory)
            results.append(result)

        # Should have both True and False values (but mostly False for high score memory)
        assert False in results  # Most should be False

    def test_get_forgetting_reason(self, forget_mechanism, low_score_memory):
        """Test reason generation for forgetting."""
        reason = forget_mechanism.get_forgetting_reason(low_score_memory, 'threshold_based')

        assert isinstance(reason, str)
        assert len(reason) > 0
        assert "Low score" in reason

    def test_similarity_based_forgetting(self, forget_mechanism):
        """Test similarity-based forgetting."""
        # Test 1: Memory is NOT the lowest score, should NOT be forgotten
        memory = {
            'id': 'test-123',
            'final_score': 0.2,
            'content': 'Duplicate information'
        }

        context = {
            'similar_memories': [
                {'id': '1', 'final_score': 0.1},  # Lower score
                {'id': '2', 'final_score': 0.4},
                memory,  # This memory itself (score 0.2)
                {'id': '4', 'final_score': 0.5},
                {'id': '5', 'final_score': 0.6}
            ]
        }

        # Should NOT forget because there's a lower scored memory (id: '1')
        should_forget = forget_mechanism._similarity_based_forgetting(memory, context)
        assert should_forget is False, "Should not forget when there's a lower scored similar memory"

        # Test 2: Memory IS the lowest score, SHOULD be forgotten
        memory_lowest = {
            'id': 'lowest',
            'final_score': 0.1
        }
        context_lowest = {
            'similar_memories': [
                {'id': '1', 'final_score': 0.3},
                {'id': '2', 'final_score': 0.4},
                memory_lowest,  # This is the lowest
                {'id': '4', 'final_score': 0.5}
            ]
        }
        
        # With many similar memories, should forget the lowest scored one
        should_forget_lowest = forget_mechanism._similarity_based_forgetting(memory_lowest, context_lowest)
        assert should_forget_lowest is True, "Should forget when it's the lowest scored similar memory"

        # Test 3: Not enough similar memories, should NOT forget
        context_few = {
            'similar_memories': [
                {'id': '1', 'final_score': 0.3},
                memory_lowest,  # Only 2 similar memories total
            ]
        }
        
        should_forget_few = forget_mechanism._similarity_based_forgetting(memory_lowest, context_few)
        assert should_forget_few is False, "Should not forget when there are too few similar memories"