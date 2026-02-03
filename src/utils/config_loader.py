import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    
    if not os.path.exists(config_path):
        # Create default config
        default_config = {
            'memory': {
                'short_term': {'capacity': 10, 'ttl_seconds': 3600},
                'long_term': {'vector_db': 'chromadb', 'collection_name': 'agent_memory'},
                'scoring': {'weights': {'recency': 0.3, 'frequency': 0.2, 'relevance': 0.4, 'task_success': 0.1}},
                'forgetting': {'short_term_threshold': 0.2, 'long_term_threshold': 0.3}
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f)
        
        return default_config
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    """Save configuration to YAML file."""
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)