"""
Data model definitions
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class AttackStrategy(Enum):
    """Attack strategy types"""
    VALUE_DECEPTION = "value_deception"  # Value deception
    ROLE_PLAY = "role_play"  # Role play
    NARRATIVE_DISGUISE = "narrative_disguise"  # Narrative disguise
    LOGIC_MANIPULATION = "logic_manipulation"  # Logic manipulation


@dataclass
class VulnerabilityNode:
    """Vulnerability node - stored in knowledge graph"""
    id: str
    strategy: str  # Attack strategy used
    harmful_query: str  # Original harmful query
    attack_prompt: str  # Attack prompt
    model_response: str  # Model response
    severity: float  # Severity level 0-1
    discovered_by: str  # Discoverer agent_id
    timestamp: float  # Timestamp
