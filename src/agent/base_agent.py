from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from ..memory.manager import MemoryManager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class BaseAgent:
    """Base autonomous agent with memory capabilities."""
    
    def __init__(self, config: Dict[str, Any], memory_manager: MemoryManager):
        self.config = config
        self.memory = memory_manager
        self.conversation_history = []
        self.task_history = []
        
    def process_query(self, query: str, task_context: str = "general") -> str:
        """Process a query using memory and generate response."""
        
        # Step 1: Retrieve relevant memories
        relevant_memories = self.memory.retrieve_memories(
            query=query,
            top_k=5,
            memory_type="both"
        )
        
        # Step 2: Generate context from memories
        context = self._build_context(relevant_memories, query)
        
        # Step 3: Generate response (simplified - replace with actual LLM)
        response = self._generate_response(query, context)
        
        # Step 4: Store new memory about this interaction
        interaction_memory = {
            'query': query,
            'response': response,
            'task_context': task_context,
            'timestamp': datetime.now().isoformat()
        }
        
        # Decide memory type based on importance
        memory_type = "long" if self._is_important_interaction(interaction_memory) else "short"
        
        memory_content = f"Q: {query}\nA: {response}"
        self.memory.add_memory(
            content=memory_content,
            memory_type=memory_type,
            task_context=task_context,
            metadata={
                'interaction_type': 'qa',
                'task_success_score': self._evaluate_response_quality(response, query)
            }
        )
        
        # Update conversation history
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now().isoformat(),
            'memories_used': [m['id'] for m in relevant_memories]
        })
        
        logger.info(f"Query processed", extra={
            'query': query,
            'response_preview': response[:50] + '...',
            'memories_used': len(relevant_memories),
            'memory_type_stored': memory_type
        })
        
        return response
    
    def _build_context(self, memories: List[Dict], query: str) -> str:
        """Build context string from relevant memories."""
        
        if not memories:
            return "No relevant memories found."
        
        context_parts = ["Relevant memories from past:"]
        
        for i, memory in enumerate(memories[:3]):  # Top 3 most relevant
            content_preview = memory['content'][:100] + '...' if len(memory['content']) > 100 else memory['content']
            score = memory.get('final_score', 0.5)
            
            context_parts.append(
                f"{i+1}. [Score: {score:.2f}] {content_preview}"
            )
        
        context_parts.append(f"\nCurrent query: {query}")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using LLM (simplified version)."""
        
        # This is a placeholder - in real implementation, call an LLM
        # For example: ollama.chat(model="llama2", messages=[...])
        
        # Simple rule-based response for demonstration
        if "hello" in query.lower() or "hi" in query.lower():
            return "Hello! I'm an AI agent with memory capabilities. How can I help you today?"
        elif "memory" in query.lower():
            stats = self.memory.get_stats()
            return f"I currently have {stats['current_short_term']} short-term and {stats['current_long_term']} long-term memories."
        elif "forg" in query.lower():  # forget or forgetting
            return "I automatically forget less useful memories based on recency, frequency, and relevance scores."
        else:
            return f"I processed your query about '{query}' using my memory system. Based on my knowledge: This seems like a general inquiry that I'll remember to improve future responses."
    
    def _is_important_interaction(self, interaction: Dict) -> bool:
        """Determine if an interaction should be stored in long-term memory."""
        
        query = interaction['query'].lower()
        
        # Criteria for importance
        important_keywords = ['important', 'remember', 'key', 'critical', 'essential']
        question_types = ['what is', 'how to', 'why is', 'explain']
        
        has_important_keyword = any(keyword in query for keyword in important_keywords)
        is_question = any(q_type in query for q_type in question_types)
        
        return has_important_keyword or is_question
    
    def _evaluate_response_quality(self, response: str, query: str) -> float:
        """Evaluate how good the response was (simplified)."""
        
        # Simple heuristics for demonstration
        score = 0.5  # Base score
        
        # Longer responses might be better for complex queries
        if len(query.split()) > 5 and len(response.split()) > 20:
            score += 0.2
        
        # Responses that acknowledge memory usage
        if "memory" in response.lower() or "remember" in response.lower():
            score += 0.1
        
        # Responses that are clear and complete
        if response.endswith('.') and len(response.split()) > 5:
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def run_task(self, task_description: str) -> Dict[str, Any]:
        """Run a complete task and track performance."""
        
        task_result = {
            'task': task_description,
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'memories_created': [],
            'memories_used': []
        }
        
        # Simulate task execution
        # In real implementation, this would be more complex
        steps = [
            f"Understanding task: {task_description}",
            "Retrieving relevant memories",
            "Executing task",
            "Storing results in memory"
        ]
        
        for step in steps:
            # Process each step as a query
            response = self.process_query(step, task_context=task_description)
            
            task_result['steps'].append({
                'step': step,
                'response': response
            })
        
        task_result['end_time'] = datetime.now().isoformat()
        task_result['success'] = True
        
        self.task_history.append(task_result)
        
        return task_result
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        
        memory_stats = self.memory.get_stats()
        
        return {
            'conversations': len(self.conversation_history),
            'tasks_completed': len(self.task_history),
            'memory_stats': memory_stats,
            'last_activity': self.conversation_history[-1]['timestamp'] if self.conversation_history else None
        }