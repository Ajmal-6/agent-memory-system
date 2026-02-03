from datetime import datetime
from typing import Dict, Any
import numpy as np


class MemoryScorer:
    """Calculate and update memory scores based on multiple factors."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = config['weights']
        self.recency_decay = config['recency_decay_rate']

    def calculate_initial_score(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate initial scores for a new memory."""

        # Initial scores
        metadata['recency_score'] = 1.0  # Brand new
        metadata['frequency_score'] = 0.1  # Small base value for new memory
        metadata['relevance_score'] = metadata.get('relevance_score', 0.5)
        metadata['task_success_score'] = metadata.get('task_success_score', 0.5)

        # Calculate final score
        metadata['final_score'] = self._calculate_weighted_score(metadata)
        metadata['score_history'] = [{
            'timestamp': metadata['created_at'],
            'score': metadata['final_score'],
            'type': 'initial'
        }]

        return metadata

    def update_score(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update scores based on new access/usage."""

        # Update access count
        metadata['access_count'] = metadata.get('access_count', 0) + 1

        # Update recency score (exponential decay)
        current_time = datetime.now()
        last_accessed = datetime.fromisoformat(metadata['last_accessed'])
        hours_passed = (current_time - last_accessed).total_seconds() / 3600

        metadata['recency_score'] = np.exp(-self.recency_decay * hours_passed)

        # Update frequency score (normalized access count)
        max_access_window = self.config['frequency_window']
        access_count = metadata['access_count']
        metadata['frequency_score'] = min(access_count / max_access_window, 1.0)

        # Update last accessed time
        metadata['last_accessed'] = current_time.isoformat()

        # Keep relevance and task success scores (they might be updated elsewhere)
        if 'relevance_score' not in metadata:
            metadata['relevance_score'] = 0.5
        if 'task_success_score' not in metadata:
            metadata['task_success_score'] = 0.5

        # Calculate new final score
        metadata['final_score'] = self._calculate_weighted_score(metadata)

        # Add to score history
        if 'score_history' not in metadata:
            metadata['score_history'] = []

        metadata['score_history'].append({
            'timestamp': current_time.isoformat(),
            'score': metadata['final_score'],
            'type': 'update'
        })

        return metadata

    def calculate_final_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate final score without updating metadata."""
        scores = {
            'recency': self._calculate_recency_score(metadata),
            'frequency': self._calculate_frequency_score(metadata),
            'relevance': metadata.get('relevance_score', 0.5),
            'task_success': metadata.get('task_success_score', 0.5)
        }

        final_score = sum(
            scores[factor] * self.weights[factor]
            for factor in self.weights
        )

        return final_score

    def _calculate_weighted_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate weighted sum of all scores."""
        scores = [
            metadata['recency_score'] * self.weights['recency'],
            metadata['frequency_score'] * self.weights['frequency'],
            metadata['relevance_score'] * self.weights['relevance'],
            metadata['task_success_score'] * self.weights['task_success']
        ]

        return sum(scores)

    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score without updating metadata."""
        current_time = datetime.now()
        last_accessed_str = metadata.get('last_accessed', metadata.get('created_at', current_time.isoformat()))
        
        try:
            last_accessed = datetime.fromisoformat(last_accessed_str)
            hours_passed = (current_time - last_accessed).total_seconds() / 3600
            return np.exp(-self.recency_decay * hours_passed)
        except Exception:
            return 0.5  # Default score if parsing fails

    def _calculate_frequency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate frequency score without updating metadata."""
        access_count = metadata.get('access_count', 0)
        max_access_window = self.config['frequency_window']

        return min(access_count / max_access_window, 1.0)