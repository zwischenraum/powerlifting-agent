import logging
import os

def setup_logger():
    """Configure logging based on DEBUG environment variable"""
    log_level = logging.DEBUG if os.getenv('DEBUG') == 'true' else logging.INFO
    
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)

logger = setup_logger()
