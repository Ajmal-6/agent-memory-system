import pytest
from datetime import datetime, timedelta
from src.memory.scoring import MemoryScorer


class TestMemoryScorer:
    @pytest.fixture
    def scorer(self):
        config = {
            'weights': {
                'recency': 0.3,
                'frequency': 0.2,
                'relevance': 0.4,
                'task_success': 0.1
            },
            'recency_decay_rate': 0.1,
            'frequency_window': 100
        }
        return MemoryScorer(config)

    @pytest.fixture
    def sample_metadata(self):
        return {
            'id': 'test-123',
            'content': 'Test content',
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 1,
            'relevance_score': 0.8,
            'task_success_score': 0.9
        }

    def test_calculate_initial_score(self, scorer, sample_metadata):
        """Test initial score calculation."""
        result = scorer.calculate_initial_score(sample_metadata)

        assert 'final_score' in result
        assert 'score_history' in result
        assert result['final_score'] > 0
        assert len(result['score_history']) == 1
        assert result['frequency_score'] == 0.1  # Initial value

    def test_update_score(self, scorer, sample_metadata):
        """Test score update after access."""
        # Initialize first
        metadata = scorer.calculate_initial_score(sample_metadata)
        initial_score = metadata['final_score']

        # Simulate time passing
        old_time = datetime.now() - timedelta(hours=10)
        metadata['last_accessed'] = old_time.isoformat()
        metadata['access_count'] = 5

        # Update score
        updated = scorer.update_score(metadata)

        # Access count should be incremented
        assert updated['access_count'] == 6
        # Score should change
        assert updated['final_score'] != initial_score
        # Should have score history
        assert len(updated['score_history']) == 2
        # Last accessed should be updated
        assert updated['last_accessed'] != old_time.isoformat()

    def test_calculate_final_score(self, scorer, sample_metadata):
        """Test final score calculation."""
        metadata = scorer.calculate_initial_score(sample_metadata)

        final_score = scorer.calculate_final_score(metadata)

        assert isinstance(final_score, float)
        assert 0 <= final_score <= 1

    def test_score_components(self, scorer, sample_metadata):
        """Test individual score components."""
        metadata = scorer.calculate_initial_score(sample_metadata)

        # Check all components exist
        assert 'recency_score' in metadata
        assert 'frequency_score' in metadata
        assert 'relevance_score' in metadata
        assert 'task_success_score' in metadata

        # All should be between 0 and 1
        assert 0 <= metadata['recency_score'] <= 1
        assert 0 <= metadata['frequency_score'] <= 1
        assert 0 <= metadata['relevance_score'] <= 1
        assert 0 <= metadata['task_success_score'] <= 1