"""
配置文件 - 自定义OpenAI格式API配置
"""

# Attacker Model Configuration (drives all agents)
ATTACKER_MODEL_CONFIG = {
    "api_key": "EMPTY",
    "base_url": "http://localhost:8000/v1",
    "model": "vicuna"
}

# Victim Model Configuration (target model for testing)
VICTIM_MODEL_CONFIG = {
    "api_key": "EMPTY", 
    "base_url": "http://localhost:8001/v1",  # Different port for victim model
    "model": "target-model"
}

# Agent interaction delay (seconds)
AGENT_INTERACTION_DELAY = 2

# Nash equilibrium parameters
NASH_EQUILIBRIUM_CONFIG = {
    "max_iterations": 100,
    "convergence_threshold": 0.001,
    "exploration_bonus": 0.1,
    "performance_weight": 0.7
}

# Knowledge sharing parameters
KNOWLEDGE_SHARING_CONFIG = {
    "credibility_threshold": 0.6,
    "trust_decay_rate": 0.05,
    "collaboration_bonus": 0.2
}

# Coverage optimization parameters
COVERAGE_CONFIG = {
    "threat_space_dimensions": 5,
    "sampling_strategy": "adaptive",
    "coverage_threshold": 0.8
}