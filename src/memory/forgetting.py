from typing import List, Dict, Any
from datetime import datetime
import random


class ForgettingMechanism:
    """Handle automated forgetting based on various strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies = ['threshold_based', 'random', 'similarity_based']

    def decide_to_forget(self,
                         memory: Dict[str, Any],
                         strategy: str = 'threshold_based',
                         context: Dict[str, Any] = None) -> bool:
        """Decide whether to forget a memory."""

        if strategy == 'threshold_based':
            return self._threshold_based_forgetting(memory)
        elif strategy == 'random':
            return self._random_forgetting(memory)
        elif strategy == 'similarity_based':
            return self._similarity_based_forgetting(memory, context)
        else:
            return self._threshold_based_forgetting(memory)

    def _threshold_based_forgetting(self, memory: Dict[str, Any]) -> bool:
        """Forget if score is below threshold."""
        final_score = memory.get('final_score', 0.5)

        if memory.get('type') == 'short':
            threshold = self.config['short_term_threshold']
        else:
            threshold = self.config['long_term_threshold']

        return final_score < threshold

    def _random_forgetting(self, memory: Dict[str, Any]) -> bool:
        """Random forgetting (can be used for exploration)."""
        age_days = self._calculate_memory_age_days(memory)

        # Older memories have higher chance of being forgotten
        forget_probability = min(0.05 + (age_days * 0.02), 0.3)

        return random.random() < forget_probability

    def _similarity_based_forgetting(self,
                                     memory: Dict[str, Any],
                                     context: Dict[str, Any]) -> bool:
        """Forget similar memories to reduce redundancy."""
        if not context or 'similar_memories' not in context:
            return False

        similar_memories = context['similar_memories']
        
        # If we have too many similar memories, forget the lowest scored one
        if len(similar_memories) > 3:  # Arbitrary threshold
            # Find the memory with lowest score among similar ones
            # Include ALL memories (including the current one) in comparison
            lowest_score = min(m.get('final_score', 0.5) for m in similar_memories)
            # Return True if this memory has the lowest score
            return memory.get('final_score', 0.5) == lowest_score
        
        return False

    def _calculate_memory_age_days(self, memory: Dict[str, Any]) -> float:
        """Calculate age of memory in days."""
        try:
            created_at_str = memory.get('created_at', datetime.now().isoformat())
            created_at = datetime.fromisoformat(created_at_str)
            current_time = datetime.now()
            return (current_time - created_at).total_seconds() / (24 * 3600)
        except Exception:
            return 0.0

    def get_forgetting_reason(self,
                              memory: Dict[str, Any],
                              strategy: str = 'threshold_based') -> str:
        """Generate a human-readable reason for forgetting."""

        final_score = memory.get('final_score', 0.5)

        if strategy == 'threshold_based':
            threshold = (self.config['short_term_threshold']
                         if memory.get('type') == 'short'
                         else self.config['long_term_threshold'])

            reasons = []
            if final_score < threshold:
                reasons.append(f"Low score ({final_score:.2f} < {threshold})")

            # Check individual component scores
            if memory.get('recency_score', 1.0) < 0.3:
                reasons.append("Not accessed recently")
            if memory.get('frequency_score', 0.0) < 0.1:
                reasons.append("Rarely used")
            if memory.get('relevance_score', 0.5) < 0.3:
                reasons.append("Low relevance")

            return "; ".join(reasons) if reasons else "Unknown reason"

        elif strategy == 'random':
            return "Random forgetting for exploration"

        elif strategy == 'similarity_based':
            return "Redundant information (similar to other memories)"

        return "Unknown reason"