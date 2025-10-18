"""
简化的数据模型
"""

from dataclasses import dataclass
from enum import Enum
from typing import List
from datetime import datetime

class AttackType(Enum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    CONTEXT_MANIPULATION = "context_manipulation"
    ADVERSARIAL_PROMPT = "adversarial_prompt"

@dataclass
class VulnerabilityReport:
    attack_type: AttackType
    prompt: str
    response: str
    severity: float  # 0-1
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()