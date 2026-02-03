import chromadb
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import deque
import uuid
import json
import os

from .scoring import MemoryScorer
from .forgetting import ForgettingMechanism
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Main memory manager handling both short-term and long-term memory."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.short_term_memory = deque(
            maxlen=config['memory']['short_term']['capacity']
        )
        self.long_term_memory = None
        self.scorer = MemoryScorer(config['memory']['scoring'])
        self.forgetting = ForgettingMechanism(config['memory']['forgetting'])
        self.operation_count = 0
        self.memory_stats = {
            'total_created': 0,
            'total_forgotten': 0,
            'current_short_term': 0,
            'current_long_term': 0
        }
        
        # Initialize storage dictionaries
        self.long_term_storage = {}
        self.mock_embeddings = {}
        self.score_histories = {}  # Store score histories separately
        
        self._init_long_term_memory()

    def _init_long_term_memory(self):
        """Initialize vector database for long-term memory."""
        db_config = self.config['memory']['long_term']

        if db_config['vector_db'] == 'chromadb':
            try:
                # Use new ChromaDB API (version >= 0.4.0)
                persist_dir = db_config.get('persist_directory')
                
                if persist_dir and os.path.exists(persist_dir):
                    # Use persistent client
                    self.chroma_client = chromadb.PersistentClient(
                        path=persist_dir
                    )
                else:
                    # Use in-memory client for testing
                    self.chroma_client = chromadb.EphemeralClient()

                # Create or get collection
                try:
                    self.collection = self.chroma_client.get_collection(
                        db_config['collection_name']
                    )
                except Exception:
                    self.collection = self.chroma_client.create_collection(
                        name=db_config['collection_name'],
                        metadata={"description": "Agent long-term memory"}
                    )

                # Initialize embeddings model
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            except Exception as e:
                logger.warning(f"ChromaDB initialization failed: {e}")
                self._init_mock_memory()
        else:
            # Use mock for testing
            self._init_mock_memory()

    def _init_mock_memory(self):
        """Initialize mock memory storage for testing."""
        self.collection = None
        self.long_term_storage = {}
        self.mock_embeddings = {}

        # Mock embedding model
        class MockEmbeddingModel:
            def encode(self, text):
                # Generate deterministic mock embeddings
                import hashlib
                hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                np.random.seed(hash_val % 2**32)
                return np.random.rand(384).tolist()

        self.embedding_model = MockEmbeddingModel()
        logger.info("Using mock memory storage (ChromaDB not available)")

    def _prepare_chroma_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB by converting complex types."""
        chroma_metadata = {}
        for key, value in metadata.items():
            # ChromaDB only accepts primitive types
            if isinstance(value, (str, int, float, bool, type(None))):
                chroma_metadata[key] = value
            elif key == 'score_history':
                # Store score history as JSON string
                chroma_metadata['score_history_json'] = json.dumps(value)
            else:
                # Convert other types to string
                chroma_metadata[key] = str(value)
        return chroma_metadata

    def _parse_chroma_metadata(self, chroma_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Parse metadata from ChromaDB, converting JSON strings back to objects."""
        metadata = {}
        for key, value in chroma_metadata.items():
            if key == 'score_history_json':
                # Parse score history from JSON string
                try:
                    metadata['score_history'] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    metadata['score_history'] = []
            elif isinstance(value, str) and value.startswith('[') and value.endswith(']'):
                # Try to parse as list if it looks like one
                try:
                    metadata[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    metadata[key] = value
            else:
                metadata[key] = value
        return metadata

    def add_memory(self,
                   content: str,
                   memory_type: str = "long",
                   metadata: Optional[Dict] = None,
                   task_context: Optional[str] = None) -> str:
        """Add a new memory to the system."""

        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        base_metadata = {
            'id': memory_id,
            'content': content,
            'type': memory_type,
            'created_at': timestamp,
            'last_accessed': timestamp,
            'access_count': 0,
            'relevance_score': 0.0,
            'task_context': task_context or "general",
        }

        if metadata:
            base_metadata.update(metadata)

        # Calculate initial score - this adds score_history
        base_metadata = self.scorer.calculate_initial_score(base_metadata)

        if memory_type == "short":
            base_metadata['ttl'] = (
                    datetime.now() +
                    timedelta(seconds=self.config['memory']['short_term']['ttl_seconds'])
            ).isoformat()
            self.short_term_memory.append(base_metadata)
            self.memory_stats['current_short_term'] += 1

        else:  # long-term memory
            # Generate embedding
            embedding = self.embedding_model.encode(content).tolist()

            # Store score history separately
            self.score_histories[memory_id] = base_metadata.get('score_history', [])

            if self.collection is not None:
                # Prepare metadata for ChromaDB
                chroma_metadata = self._prepare_chroma_metadata(base_metadata)
                
                try:
                    self.collection.add(
                        ids=[memory_id],
                        embeddings=[embedding],
                        metadatas=[chroma_metadata],
                        documents=[content]
                    )
                except Exception as e:
                    logger.error(f"Failed to add to ChromaDB: {e}")
                    # Fallback to mock storage
                    self.long_term_storage[memory_id] = base_metadata
                    self.mock_embeddings[memory_id] = embedding
            else:
                # Mock storage
                self.long_term_storage[memory_id] = base_metadata
                self.mock_embeddings[memory_id] = embedding

            self.memory_stats['current_long_term'] += 1

        self.memory_stats['total_created'] += 1
        self.operation_count += 1

        logger.info("Memory added", extra={
            'memory_id': memory_id,
            'type': memory_type,
            'content_preview': content[:50] + '...' if len(content) > 50 else content,
            'operation': 'add'
        })

        # Periodic cleanup
        if self.operation_count % self.config['memory']['forgetting']['cleanup_interval'] == 0:
            self._cleanup_memories()

        return memory_id

    def retrieve_memories(self,
                          query: str,
                          top_k: int = 5,
                          memory_type: str = "both") -> List[Dict]:
        """Retrieve relevant memories based on query."""

        results = []

        # Search in short-term memory
        if memory_type in ["both", "short"]:
            short_term_results = []
            for memory in self.short_term_memory:
                similarity = self._calculate_similarity(query, memory['content'])
                memory['relevance_score'] = similarity
                short_term_results.append(memory)

            # Sort by relevance
            short_term_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            results.extend(short_term_results[:top_k])

            # Update access stats for short-term memories
            for memory in short_term_results[:top_k]:
                memory['last_accessed'] = datetime.now().isoformat()
                memory['access_count'] += 1
                memory = self.scorer.update_score(memory)

        # Search in long-term memory
        if memory_type in ["both", "long"]:
            query_embedding = self.embedding_model.encode(query).tolist()

            if self.collection is not None:
                # Real ChromaDB search
                try:
                    long_term_results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k,
                        include=['metadatas', 'documents', 'distances']
                    )

                    # Process long-term results
                    if long_term_results['metadatas']:
                        for i, (chroma_metadata, content, distance) in enumerate(zip(
                                long_term_results['metadatas'][0],
                                long_term_results['documents'][0],
                                long_term_results['distances'][0] if long_term_results['distances'] else [0.0] *
                                len(long_term_results['metadatas'][0])
                        )):
                            if chroma_metadata:  # Check if metadata exists
                                # Parse ChromaDB metadata
                                metadata = self._parse_chroma_metadata(chroma_metadata)
                                
                                # Get score history from separate storage
                                memory_id = metadata.get('id')
                                if memory_id and memory_id in self.score_histories:
                                    metadata['score_history'] = self.score_histories[memory_id]
                                
                                metadata['content'] = content
                                metadata['relevance_score'] = 1 - distance
                                metadata['last_accessed'] = datetime.now().isoformat()
                                metadata['access_count'] += 1
                                metadata = self.scorer.update_score(metadata)
                                
                                # Update score history storage
                                if memory_id:
                                    self.score_histories[memory_id] = metadata.get('score_history', [])
                                
                                # Update in ChromaDB with processed metadata
                                updated_chroma_metadata = self._prepare_chroma_metadata(metadata)
                                try:
                                    self.collection.update(
                                        ids=[metadata['id']],
                                        metadatas=[updated_chroma_metadata]
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to update ChromaDB: {e}")

                                results.append(metadata)
                except Exception as e:
                    logger.error(f"ChromaDB query failed: {e}")
                    # Fallback to mock search
                    results.extend(self._mock_search(query_embedding, top_k))
            else:
                # Mock search
                results.extend(self._mock_search(query_embedding, top_k))

        # Sort all results by relevance
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        logger.info("Memories retrieved", extra={
            'query': query,
            'num_results': len(results),
            'memory_type': memory_type,
            'operation': 'retrieve'
        })

        return results[:top_k]

    def _mock_search(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Mock search for long-term memories."""
        import numpy as np

        similarities = []
        for memory_id, embedding in self.mock_embeddings.items():
            if memory_id in self.long_term_storage:
                vec1 = np.array(query_embedding)
                vec2 = np.array(embedding)
                # Cosine similarity
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
                else:
                    similarity = 0.0
                similarities.append((memory_id, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top k memories
        results = []
        for memory_id, similarity in similarities[:top_k]:
            memory = self.long_term_storage[memory_id].copy()
            memory['relevance_score'] = float(similarity)
            memory['last_accessed'] = datetime.now().isoformat()
            memory['access_count'] += 1
            memory = self.scorer.update_score(memory)
            
            # Update the memory in long_term_storage
            self.long_term_storage[memory_id] = memory
            
            results.append(memory)

        return results

    def forget_memory(self, memory_id: str, reason: str = "low_score"):
        """Forget/delete a specific memory."""

        # Try to find in short-term memory
        for i, memory in enumerate(self.short_term_memory):
            if memory['id'] == memory_id:
                del self.short_term_memory[i]
                self.memory_stats['current_short_term'] -= 1
                self.memory_stats['total_forgotten'] += 1

                logger.info("Short-term memory forgotten", extra={
                    'memory_id': memory_id,
                    'reason': reason,
                    'operation': 'forget'
                })
                return True

        # Try to delete from long-term memory
        if self.collection is not None:
            try:
                self.collection.delete(ids=[memory_id])
                self.memory_stats['current_long_term'] -= 1
                self.memory_stats['total_forgotten'] += 1
                
                # Clean up score history
                if memory_id in self.score_histories:
                    del self.score_histories[memory_id]

                logger.info("Long-term memory forgotten", extra={
                    'memory_id': memory_id,
                    'reason': reason,
                    'operation': 'forget'
                })
                return True
            except Exception as e:
                logger.warning(f"Failed to delete from ChromaDB: {e}")

        # Try mock storage
        if memory_id in self.long_term_storage:
            del self.long_term_storage[memory_id]
            if memory_id in self.mock_embeddings:
                del self.mock_embeddings[memory_id]
            # Clean up score history
            if memory_id in self.score_histories:
                del self.score_histories[memory_id]
                
            self.memory_stats['current_long_term'] -= 1
            self.memory_stats['total_forgotten'] += 1

            logger.info("Long-term memory forgotten (mock)", extra={
                'memory_id': memory_id,
                'reason': reason,
                'operation': 'forget'
            })
            return True

        return False

    def _cleanup_memories(self):
        """Automatic cleanup of memories based on scoring."""
        forgotten_count = 0

        # Clean short-term memories
        current_time = datetime.now()
        to_remove = []

        for i, memory in enumerate(self.short_term_memory):
            # Check TTL
            if 'ttl' in memory:
                ttl_time = datetime.fromisoformat(memory['ttl'])
                if current_time > ttl_time:
                    to_remove.append(i)
                    continue

            # Check score threshold
            final_score = self.scorer.calculate_final_score(memory)
            if final_score < self.config['memory']['forgetting']['short_term_threshold']:
                to_remove.append(i)

        # Remove in reverse order
        for i in sorted(to_remove, reverse=True):
            memory_id = self.short_term_memory[i]['id']
            self.short_term_memory.pop(i)
            self.memory_stats['current_short_term'] -= 1
            self.memory_stats['total_forgotten'] += 1
            forgotten_count += 1

            logger.info("Auto-forgotten short-term memory", extra={
                'memory_id': memory_id,
                'reason': 'low_score_or_ttl',
                'operation': 'auto_forget'
            })

        # Clean long-term memories (batch processing)
        batch_size = self.config['memory']['forgetting']['batch_size']

        if self.collection is not None:
            # Real ChromaDB cleanup
            try:
                all_memories = self.collection.get(include=['metadatas'])
                to_forget = []
                for chroma_metadata in all_memories['metadatas']:
                    # Parse metadata
                    metadata = self._parse_chroma_metadata(chroma_metadata)
                    # Get score history
                    memory_id = metadata.get('id')
                    if memory_id and memory_id in self.score_histories:
                        metadata['score_history'] = self.score_histories[memory_id]
                    
                    final_score = self.scorer.calculate_final_score(metadata)
                    if final_score < self.config['memory']['forgetting']['long_term_threshold']:
                        to_forget.append(memory_id)

                        if len(to_forget) >= batch_size:
                            break

                # Delete batch
                if to_forget:
                    self.collection.delete(ids=to_forget)
                    # Clean up score histories
                    for memory_id in to_forget:
                        if memory_id in self.score_histories:
                            del self.score_histories[memory_id]
                    
                    self.memory_stats['current_long_term'] -= len(to_forget)
                    self.memory_stats['total_forgotten'] += len(to_forget)
                    forgotten_count += len(to_forget)

                    logger.info("Auto-forgotten long-term memories", extra={
                        'memory_ids': to_forget,
                        'count': len(to_forget),
                        'reason': 'low_score',
                        'operation': 'auto_forget'
                    })
            except Exception as e:
                logger.error(f"ChromaDB cleanup failed: {e}")
        else:
            # Mock storage cleanup
            to_forget = []
            for memory_id, memory in list(self.long_term_storage.items())[:batch_size]:
                final_score = self.scorer.calculate_final_score(memory)
                if final_score < self.config['memory']['forgetting']['long_term_threshold']:
                    to_forget.append(memory_id)

            # Delete from mock storage
            for memory_id in to_forget:
                if memory_id in self.long_term_storage:
                    del self.long_term_storage[memory_id]
                if memory_id in self.mock_embeddings:
                    del self.mock_embeddings[memory_id]
                # Clean up score history
                if memory_id in self.score_histories:
                    del self.score_histories[memory_id]
                    
                self.memory_stats['current_long_term'] -= 1
                self.memory_stats['total_forgotten'] += 1
                forgotten_count += 1

            if to_forget:
                logger.info("Auto-forgotten long-term memories (mock)", extra={
                    'memory_ids': to_forget,
                    'count': len(to_forget),
                    'reason': 'low_score',
                    'operation': 'auto_forget'
                })

        return forgotten_count

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (can be improved)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        return {
            **self.memory_stats,
            'operation_count': self.operation_count,
            'short_term_capacity': len(self.short_term_memory),
            'score_histories_count': len(self.score_histories),
            'config': {
                'forgetting_threshold_short': self.config['memory']['forgetting']['short_term_threshold'],
                'forgetting_threshold_long': self.config['memory']['forgetting']['long_term_threshold']
            }
        }

    def export_logs(self, filepath: str = "./logs/memory_export.json"):
        """Export memory logs for analysis."""
        logs = {
            'stats': self.get_stats(),
            'short_term_memories': list(self.short_term_memory),
            'score_histories': self.score_histories,
            'config': self.config
        }

        with open(filepath, 'w') as f:
            json.dump(logs, f, indent=2, default=str)

        return filepath