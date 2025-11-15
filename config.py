"""
Configuration file
"""

# Attacker Model - drives all agent reasoning
ATTACKER_MODEL_CONFIG = {
    "api_key": "EMPTY",
    "base_url": "http://192.168.147.8:8000/v1",
    "model": "deepseek"
}

# Victim Model - target model for testing
VICTIM_MODEL_CONFIG = {
    "api_key": "EMPTY",
    "base_url": "http://192.168.147.8:8000/v1",
    "model": "deepseek"
}

# Agent interaction delay (seconds)
AGENT_DELAY = 1.5
