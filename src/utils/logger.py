import logging
import json
from datetime import datetime
from typing import Dict, Any
import sys

class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if they exist
        if hasattr(record, 'extra'):
            log_object.update(record.extra)
        
        return json.dumps(log_object)

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a configured logger instance."""
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, level))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler for JSON logs
        file_handler = logging.FileHandler('./logs/agent_memory.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

def log_memory_event(logger: logging.Logger, 
                    event_type: str, 
                    memory_id: str, 
                    details: Dict[str, Any]):
    """Helper function to log memory events."""
    
    logger.info(f"Memory {event_type}", extra={
        'memory_id': memory_id,
        'event_type': event_type,
        **details
    })