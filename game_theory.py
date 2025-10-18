"""
Game Theory Algorithms for Multi-Agent Resource Allocation
Based on algorithm_design_document.md
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from loguru import logger

@dataclass
class AgentPerformance:
    """Agent performance metrics"""
    agent_id: str
    success_rate: float = 0.0
    novelty_score: float = 0.0
    collaboration_score: float = 0.0
    resource_efficiency: float = 0.5
    trust_score: float = 0.5
    vulnerabilities_found: int = 0

class NashEquilibriumSolver:
    """Nash Equilibrium Resource Allocation Algorithm"""
    
    def __init__(self, max_iterations: int = 100, convergence_threshold: float = 0.001):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
    def solve_allocation(self, agents: List[AgentPerformance], total_resources: float = 1.0) -> Dict[str, float]:
        """
        Solve Nash equilibrium for resource allocation
        Based on algorithm_design_document.md Section 1.2
        """
        if not agents:
            return {}
            
        n_agents = len(agents)
        
        # Build payoff matrix
        payoff_matrix = self._build_payoff_matrix(agents)
        
        # Initialize allocation
        allocation = {agent.agent_id: 1.0 / max(n_agents, 1) for agent in agents}
        
        logger.info(f"Starting Nash equilibrium solver with {n_agents} agents")
        
        for iteration in range(self.max_iterations):
            old_allocation = allocation.copy()
            
            # Calculate best response for each agent
            for i, agent in enumerate(agents):
                best_response = self._calculate_best_response(
                    agent, agents, allocation, payoff_matrix
                )
                allocation[agent.agent_id] = best_response
            
            # Normalize allocation
            allocation = self._normalize_allocation(allocation, total_resources)
            
            # Check convergence
            if self._is_converged(old_allocation, allocation):
                logger.info(f"Nash equilibrium converged after {iteration + 1} iterations")
                break
        
        return allocation
    
    def _build_payoff_matrix(self, agents: List[AgentPerformance]) -> np.ndarray:
        """Build payoff matrix for agents"""
        n_agents = len(agents)
        payoff_matrix = np.zeros((n_agents, n_agents))
        
        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                if i == j:
                    # Self payoff based on individual performance
                    payoff = (
                        0.4 * agent_i.success_rate +
                        0.3 * agent_i.novelty_score +
                        0.2 * agent_i.resource_efficiency +
                        0.1 * agent_i.trust_score
                    )
                else:
                    # Interaction payoff based on collaboration potential
                    collaboration_potential = (
                        agent_i.collaboration_score * agent_j.collaboration_score
                    )
                    payoff = 0.1 * collaboration_potential
                
                payoff_matrix[i][j] = payoff
        
        return payoff_matrix
    
    def _calculate_best_response(self, agent: AgentPerformance, all_agents: List[AgentPerformance], 
                                current_allocation: Dict[str, float], payoff_matrix: np.ndarray) -> float:
        """Calculate best response for an agent"""
        agent_idx = next(i for i, a in enumerate(all_agents) if a.agent_id == agent.agent_id)
        
        # Calculate utility for different allocation levels
        other_total = sum(
            current_allocation[other_agent.agent_id] 
            for other_agent in all_agents 
            if other_agent.agent_id != agent.agent_id
        )
        
        remaining_resources = 1.0 - other_total
        
        # Optimal allocation based on marginal utility
        base_utility = payoff_matrix[agent_idx][agent_idx]
        exploration_bonus = 0.1 if agent.vulnerabilities_found < 5 else 0
        
        optimal_allocation = min(
            max(base_utility + exploration_bonus, 0.05),  # Minimum 5%
            min(remaining_resources, 0.5)  # Maximum 50%
        )
        
        return optimal_allocation
    
    def _normalize_allocation(self, allocation: Dict[str, float], total_resources: float) -> Dict[str, float]:
        """Normalize allocation to sum to total_resources"""
        current_total = sum(allocation.values())
        if current_total == 0:
            return allocation
            
        normalized = {
            agent_id: (alloc / max(current_total, 1e-10)) * total_resources
            for agent_id, alloc in allocation.items()
        }
        
        return normalized
    
    def _is_converged(self, old_allocation: Dict[str, float], new_allocation: Dict[str, float]) -> bool:
        """Check if allocation has converged"""
        for agent_id in old_allocation:
            if abs(old_allocation[agent_id] - new_allocation[agent_id]) > self.convergence_threshold:
                return False
        return True

class TrustCalculator:
    """Calculate agent trust scores based on interaction history"""
    
    def __init__(self, trust_decay_rate: float = 0.05):
        self.trust_decay_rate = trust_decay_rate
    
    def calculate_trust(self, agent_id: str, interaction_history: List[Dict[str, Any]]) -> float:
        """
        Calculate agent trust score
        Based on algorithm_design_document.md Section 3.1
        """
        if not interaction_history:
            return 0.5  # Default trust
        
        # Filter interactions involving this agent
        agent_interactions = [
            interaction for interaction in interaction_history
            if interaction.get('agent_id') == agent_id
        ]
        
        if not agent_interactions:
            return 0.5
        
        # Calculate success rate
        successful_interactions = sum(
            1 for interaction in agent_interactions
            if interaction.get('success', False)
        )
        success_rate = successful_interactions / max(len(agent_interactions), 1)
        
        # Calculate novelty score
        novelty_score = self._calculate_novelty_score(agent_interactions)
        
        # Calculate collaboration score
        collaboration_score = self._calculate_collaboration_score(agent_interactions)
        
        # Composite trust score
        trust_score = (
            0.4 * success_rate +
            0.3 * novelty_score +
            0.3 * collaboration_score
        )
        
        return min(max(trust_score, 0.0), 1.0)
    
    def _calculate_novelty_score(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate novelty score based on unique discoveries"""
        if not interactions:
            return 0.0
        
        unique_discoveries = set()
        for interaction in interactions:
            if interaction.get('vulnerability_id'):
                unique_discoveries.add(interaction['vulnerability_id'])
        
        # Normalize by total interactions
        novelty_score = len(unique_discoveries) / max(len(interactions), 1)
        return min(novelty_score, 1.0)
    
    def _calculate_collaboration_score(self, interactions: List[Dict[str, Any]]) -> float:
        """Calculate collaboration score based on knowledge sharing"""
        if not interactions:
            return 0.0
        
        collaboration_events = sum(
            1 for interaction in interactions
            if interaction.get('shared_knowledge', False)
        )
        
        collaboration_score = collaboration_events / max(len(interactions), 1)
        return min(collaboration_score, 1.0)

class SwarmIntelligenceOptimizer:
    """
    Swarm Intelligence Algorithm for Strategy Optimization
    Based on algorithm_design_document.md Section 3.2
    """
    
    def __init__(self, max_iterations: int = 50, inertia: float = 0.9, 
                 cognitive_weight: float = 2.0, social_weight: float = 2.0):
        self.max_iterations = max_iterations
        self.inertia = inertia
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
    
    def optimize_strategies(self, agents: List[AgentPerformance], 
                          target_function: callable) -> Dict[str, Any]:
        """
        Optimize agent strategies using particle swarm optimization
        """
        n_agents = len(agents)
        if n_agents == 0:
            return {}
        
        # Initialize particles (agent strategies)
        particles = [self._initialize_strategy() for _ in range(n_agents)]
        velocities = [self._initialize_velocity() for _ in range(n_agents)]
        
        personal_best = particles.copy()
        personal_best_scores = [target_function(p) for p in particles]
        
        global_best_idx = np.argmax(personal_best_scores)
        global_best = personal_best[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        logger.info(f"Starting swarm optimization with {n_agents} particles")
        
        for iteration in range(self.max_iterations):
            for i in range(n_agents):
                # Update velocity
                r1, r2 = np.random.random(2)
                
                cognitive_component = (
                    self.cognitive_weight * r1 * 
                    (personal_best[i] - particles[i])
                )
                
                social_component = (
                    self.social_weight * r2 * 
                    (global_best - particles[i])
                )
                
                velocities[i] = (
                    self.inertia * velocities[i] + 
                    cognitive_component + 
                    social_component
                )
                
                # Update position (strategy)
                particles[i] = particles[i] + velocities[i]
                particles[i] = self._apply_constraints(particles[i])
                
                # Evaluate fitness
                fitness = target_function(particles[i])
                
                # Update personal best
                if fitness > personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = fitness
                    
                    # Update global best
                    if fitness > global_best_score:
                        global_best = particles[i]
                        global_best_score = fitness
        
        logger.info(f"Swarm optimization completed. Best score: {global_best_score:.4f}")
        
        return {
            'best_strategy': global_best,
            'best_score': global_best_score,
            'all_strategies': particles
        }
    
    def _initialize_strategy(self) -> np.ndarray:
        """Initialize random strategy vector"""
        return np.random.uniform(-1, 1, 10)  # 10-dimensional strategy space
    
    def _initialize_velocity(self) -> np.ndarray:
        """Initialize random velocity vector"""
        return np.random.uniform(-0.1, 0.1, 10)
    
    def _apply_constraints(self, strategy: np.ndarray) -> np.ndarray:
        """Apply constraints to strategy vector"""
        return np.clip(strategy, -2, 2)  # Constrain to [-2, 2] range