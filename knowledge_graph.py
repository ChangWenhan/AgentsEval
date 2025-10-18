"""
Knowledge Graph and Information Sharing System
Based on algorithm_design_document.md
"""

import json
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from loguru import logger

@dataclass
class VulnerabilityNode:
    """Vulnerability node in knowledge graph"""
    id: str
    attack_type: str
    severity: float
    prompt: str
    response: str
    success_indicators: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    source_agent: str = ""
    credibility: float = 0.5

@dataclass
class AttackPattern:
    """Attack pattern extracted from vulnerabilities"""
    pattern_id: str
    attack_type: str
    template: str
    success_rate: float
    obfuscation_techniques: List[str] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)

class KnowledgeGraph:
    """
    Distributed Knowledge Graph for Vulnerability Management
    Based on algorithm_design_document.md Section 1.2
    """
    
    def __init__(self):
        self.vulnerabilities: Dict[str, VulnerabilityNode] = {}
        self.attack_patterns: Dict[str, AttackPattern] = {}
        self.correlation_matrix: np.ndarray = np.array([])
        self.agent_contributions: Dict[str, List[str]] = defaultdict(list)
        self.trust_scores: Dict[str, float] = defaultdict(lambda: 0.5)
        
    def add_vulnerability(self, vulnerability: VulnerabilityNode, source_agent: str) -> None:
        """Add new vulnerability to knowledge graph"""
        vulnerability.source_agent = source_agent
        vulnerability.credibility = self.trust_scores[source_agent]
        
        self.vulnerabilities[vulnerability.id] = vulnerability
        self.agent_contributions[source_agent].append(vulnerability.id)
        
        # Update attack patterns
        self._update_attack_patterns(vulnerability)
        
        # Update correlation matrix
        self._update_correlation_matrix()
        
        logger.info(f"Added vulnerability {vulnerability.id} from agent {source_agent}")
    
    def get_relevant_knowledge(self, agent_id: str, attack_type: str) -> Dict[str, Any]:
        """
        Get relevant knowledge for specific agent and attack type
        Based on algorithm_design_document.md Section 3.1
        """
        # Filter vulnerabilities by attack type
        relevant_vulnerabilities = [
            vuln for vuln in self.vulnerabilities.values()
            if vuln.attack_type == attack_type and vuln.credibility > 0.6
        ]
        
        # Get attack patterns
        relevant_patterns = [
            pattern for pattern in self.attack_patterns.values()
            if pattern.attack_type == attack_type
        ]
        
        # Get collaboration recommendations
        collaboration_recommendations = self._get_collaboration_recommendations(agent_id)
        
        return {
            'vulnerabilities': [self._vulnerability_to_dict(v) for v in relevant_vulnerabilities],
            'attack_patterns': [self._pattern_to_dict(p) for p in relevant_patterns],
            'collaboration_recommendations': collaboration_recommendations,
            'trust_scores': dict(self.trust_scores)
        }
    
    def update_trust_scores(self, agent_trust_scores: Dict[str, float]) -> None:
        """Update agent trust scores"""
        self.trust_scores.update(agent_trust_scores)
        
        # Recalculate vulnerability credibility
        for vuln in self.vulnerabilities.values():
            if vuln.source_agent in agent_trust_scores:
                vuln.credibility = agent_trust_scores[vuln.source_agent]
    
    def extract_attack_chains(self, vulnerability_cluster: List[str]) -> List[List[str]]:
        """
        Extract attack chains from vulnerability clusters
        Based on algorithm_design_document.md Section 4.2
        """
        if not vulnerability_cluster:
            return []
        
        # Get vulnerabilities in cluster
        cluster_vulns = [
            self.vulnerabilities[vuln_id] for vuln_id in vulnerability_cluster
            if vuln_id in self.vulnerabilities
        ]
        
        if len(cluster_vulns) < 2:
            return [[vuln.id] for vuln in cluster_vulns]
        
        # Build correlation graph for cluster
        cluster_correlations = self._calculate_cluster_correlations(cluster_vulns)
        
        # Find attack chains using graph traversal
        attack_chains = self._find_attack_chains(cluster_vulns, cluster_correlations)
        
        return attack_chains
    
    def get_coverage_analysis(self) -> Dict[str, Any]:
        """
        Analyze threat space coverage
        Based on algorithm_design_document.md Section 4.1
        """
        # Define threat space dimensions
        attack_types = set(vuln.attack_type for vuln in self.vulnerabilities.values())
        severity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        # Calculate coverage for each dimension
        coverage_by_attack_type = {}
        for attack_type in attack_types:
            type_vulns = [v for v in self.vulnerabilities.values() if v.attack_type == attack_type]
            coverage_by_attack_type[attack_type] = len(type_vulns)
        
        # Calculate severity distribution
        severity_distribution = {}
        for level in severity_levels:
            count = sum(
                1 for vuln in self.vulnerabilities.values()
                if abs(vuln.severity - level) < 0.1
            )
            severity_distribution[f"severity_{level}"] = count
        
        # Identify uncovered regions
        uncovered_regions = self._identify_uncovered_regions()
        
        total_coverage = len(self.vulnerabilities) / max((len(attack_types) * len(severity_levels)), 1)
        
        return {
            'total_coverage': min(total_coverage, 1.0),
            'coverage_by_attack_type': coverage_by_attack_type,
            'severity_distribution': severity_distribution,
            'uncovered_regions': uncovered_regions,
            'total_vulnerabilities': len(self.vulnerabilities)
        }
    
    def _update_attack_patterns(self, vulnerability: VulnerabilityNode) -> None:
        """Update attack patterns based on new vulnerability"""
        # Extract pattern from vulnerability
        pattern_template = self._extract_pattern_template(vulnerability.prompt)
        
        pattern_id = f"{vulnerability.attack_type}_{len(self.attack_patterns)}"
        
        # Check if similar pattern exists
        existing_pattern = self._find_similar_pattern(pattern_template, vulnerability.attack_type)
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.success_rate = (
                existing_pattern.success_rate * 0.8 + 
                (1.0 if vulnerability.severity > 0.5 else 0.0) * 0.2
            )
        else:
            # Create new pattern
            new_pattern = AttackPattern(
                pattern_id=pattern_id,
                attack_type=vulnerability.attack_type,
                template=pattern_template,
                success_rate=1.0 if vulnerability.severity > 0.5 else 0.0,
                obfuscation_techniques=self._extract_obfuscation_techniques(vulnerability.prompt)
            )
            self.attack_patterns[pattern_id] = new_pattern
    
    def _update_correlation_matrix(self) -> None:
        """Update vulnerability correlation matrix"""
        n_vulns = len(self.vulnerabilities)
        if n_vulns == 0:
            return
        
        self.correlation_matrix = np.zeros((n_vulns, n_vulns))
        vuln_list = list(self.vulnerabilities.values())
        
        for i, vuln_i in enumerate(vuln_list):
            for j, vuln_j in enumerate(vuln_list):
                if i != j:
                    correlation = self._calculate_vulnerability_correlation(vuln_i, vuln_j)
                    self.correlation_matrix[i][j] = correlation
    
    def _calculate_vulnerability_correlation(self, vuln_a: VulnerabilityNode, 
                                           vuln_b: VulnerabilityNode) -> float:
        """Calculate correlation between two vulnerabilities"""
        # Attack type similarity
        type_similarity = 1.0 if vuln_a.attack_type == vuln_b.attack_type else 0.0
        
        # Severity similarity
        severity_similarity = 1.0 - abs(vuln_a.severity - vuln_b.severity)
        
        # Prompt similarity (simple word overlap)
        prompt_similarity = self._calculate_text_similarity(vuln_a.prompt, vuln_b.prompt)
        
        # Response similarity
        response_similarity = self._calculate_text_similarity(vuln_a.response, vuln_b.response)
        
        # Weighted correlation
        correlation = (
            0.3 * type_similarity +
            0.2 * severity_similarity +
            0.3 * prompt_similarity +
            0.2 * response_similarity
        )
        
        return correlation
    
    def _calculate_text_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate simple text similarity based on word overlap"""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)
        
        return len(intersection) / max(len(union), 1) if union else 0.0
    
    def _extract_pattern_template(self, prompt: str) -> str:
        """Extract pattern template from prompt"""
        # Simple template extraction - replace specific content with placeholders
        template = prompt
        
        # Replace common patterns
        replacements = [
            (r'\b\w+@\w+\.\w+\b', '[EMAIL]'),
            (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),
            (r'\b\d+\b', '[NUMBER]'),
        ]
        
        import re
        for pattern, replacement in replacements:
            template = re.sub(pattern, replacement, template)
        
        return template
    
    def _extract_obfuscation_techniques(self, prompt: str) -> List[str]:
        """Extract obfuscation techniques used in prompt"""
        techniques = []
        
        # Check for character substitution
        if any(char in prompt for char in ['@', '3', '1', '0', '$']):
            techniques.append('character_substitution')
        
        # Check for encoding
        if 'base64' in prompt.lower() or 'decode' in prompt.lower():
            techniques.append('encoding_obfuscation')
        
        # Check for context shifting
        context_indicators = ['hypothetical', 'roleplay', 'pretend', 'imagine']
        if any(indicator in prompt.lower() for indicator in context_indicators):
            techniques.append('context_shifting')
        
        return techniques
    
    def _find_similar_pattern(self, template: str, attack_type: str) -> AttackPattern:
        """Find similar existing pattern"""
        for pattern in self.attack_patterns.values():
            if (pattern.attack_type == attack_type and 
                self._calculate_text_similarity(pattern.template, template) > 0.7):
                return pattern
        return None
    
    def _get_collaboration_recommendations(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get collaboration recommendations for agent"""
        recommendations = []
        
        # Find agents with complementary strengths
        for other_agent, trust_score in self.trust_scores.items():
            if other_agent != agent_id and trust_score > 0.6:
                # Calculate collaboration potential
                shared_discoveries = len(
                    set(self.agent_contributions[agent_id]).intersection(
                        set(self.agent_contributions[other_agent])
                    )
                )
                
                if shared_discoveries > 0:
                    recommendations.append({
                        'agent_id': other_agent,
                        'trust_score': trust_score,
                        'shared_discoveries': shared_discoveries,
                        'collaboration_potential': trust_score * (1 + shared_discoveries * 0.1)
                    })
        
        # Sort by collaboration potential
        recommendations.sort(key=lambda x: x['collaboration_potential'], reverse=True)
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _calculate_cluster_correlations(self, cluster_vulns: List[VulnerabilityNode]) -> np.ndarray:
        """Calculate correlations within vulnerability cluster"""
        n = len(cluster_vulns)
        correlations = np.zeros((n, n))
        
        for i, vuln_i in enumerate(cluster_vulns):
            for j, vuln_j in enumerate(cluster_vulns):
                if i != j:
                    correlations[i][j] = self._calculate_vulnerability_correlation(vuln_i, vuln_j)
        
        return correlations
    
    def _find_attack_chains(self, cluster_vulns: List[VulnerabilityNode], 
                           correlations: np.ndarray) -> List[List[str]]:
        """Find attack chains using graph traversal"""
        n = len(cluster_vulns)
        chains = []
        
        # Use simple threshold-based approach
        threshold = 0.5
        
        for i in range(n):
            chain = [cluster_vulns[i].id]
            visited = {i}
            
            # Follow correlation links
            current = i
            while True:
                next_node = -1
                max_correlation = threshold
                
                for j in range(n):
                    if j not in visited and correlations[current][j] > max_correlation:
                        max_correlation = correlations[current][j]
                        next_node = j
                
                if next_node == -1:
                    break
                
                chain.append(cluster_vulns[next_node].id)
                visited.add(next_node)
                current = next_node
            
            if len(chain) > 1:
                chains.append(chain)
        
        return chains
    
    def _identify_uncovered_regions(self) -> List[Dict[str, Any]]:
        """Identify uncovered regions in threat space"""
        uncovered = []
        
        # Define expected attack types
        expected_attack_types = ['prompt_injection', 'jailbreak', 'context_manipulation', 'adversarial_prompt']
        covered_types = set(vuln.attack_type for vuln in self.vulnerabilities.values())
        
        for attack_type in expected_attack_types:
            if attack_type not in covered_types:
                uncovered.append({
                    'type': 'missing_attack_type',
                    'attack_type': attack_type,
                    'priority': 0.8
                })
        
        # Check severity coverage
        severity_ranges = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
        for low, high in severity_ranges:
            count = sum(
                1 for vuln in self.vulnerabilities.values()
                if low <= vuln.severity < high
            )
            if count == 0:
                uncovered.append({
                    'type': 'missing_severity_range',
                    'severity_range': f"{low}-{high}",
                    'priority': 0.6
                })
        
        return uncovered
    
    def _vulnerability_to_dict(self, vuln: VulnerabilityNode) -> Dict[str, Any]:
        """Convert vulnerability to dictionary"""
        return {
            'id': vuln.id,
            'attack_type': vuln.attack_type,
            'severity': vuln.severity,
            'prompt': vuln.prompt,
            'response': vuln.response,
            'success_indicators': vuln.success_indicators,
            'credibility': vuln.credibility,
            'source_agent': vuln.source_agent
        }
    
    def _pattern_to_dict(self, pattern: AttackPattern) -> Dict[str, Any]:
        """Convert attack pattern to dictionary"""
        return {
            'pattern_id': pattern.pattern_id,
            'attack_type': pattern.attack_type,
            'template': pattern.template,
            'success_rate': pattern.success_rate,
            'obfuscation_techniques': pattern.obfuscation_techniques,
            'context_requirements': pattern.context_requirements
        }