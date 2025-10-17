"""
Multi-Agent LLM Security Testing Framework - Knowledge Base and Resource Management
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque
from dataclasses import asdict
import numpy as np
from loguru import logger

from models import (
    VulnerabilityReport, AttackType, AgentRole, AgentPerformance,
    EvasionTechnique, AttackContext
)

class KnowledgePattern:
    """知识模式类"""
    
    def __init__(self, pattern_id: str, attack_type: AttackType, 
                 success_rate: float, pattern_data: Dict[str, Any]):
        self.pattern_id = pattern_id
        self.attack_type = attack_type
        self.success_rate = success_rate
        self.pattern_data = pattern_data
        self.usage_count = 0
        self.last_used = datetime.now()
        self.effectiveness_history = []
    
    def update_effectiveness(self, success: bool):
        """更新模式有效性"""
        self.usage_count += 1
        self.effectiveness_history.append(success)
        self.last_used = datetime.now()
        
        # 计算滑动窗口成功率
        recent_history = self.effectiveness_history[-20:]  # 最近20次
        self.success_rate = sum(recent_history) / len(recent_history)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "pattern_id": self.pattern_id,
            "attack_type": self.attack_type.value,
            "success_rate": self.success_rate,
            "pattern_data": self.pattern_data,
            "usage_count": self.usage_count,
            "last_used": self.last_used.isoformat(),
            "effectiveness_history": self.effectiveness_history[-10:]  # 只保存最近10次
        }

class SharedKnowledgeBase:
    """共享知识库"""
    
    def __init__(self, max_patterns: int = 1000, max_vulnerabilities: int = 5000):
        self.max_patterns = max_patterns
        self.max_vulnerabilities = max_vulnerabilities
        
        # 核心数据结构
        self.vulnerability_patterns: Dict[AttackType, List[KnowledgePattern]] = defaultdict(list)
        self.successful_strategies: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.model_responses: Dict[str, Dict[str, Any]] = {}
        self.evasion_techniques: Dict[EvasionTechnique, float] = {}
        self.agent_collaborations: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # 漏洞数据库
        self.vulnerabilities: List[VulnerabilityReport] = []
        self.vulnerability_index: Dict[str, VulnerabilityReport] = {}
        
        # 统计信息
        self.access_stats: Dict[str, int] = defaultdict(int)
        self.update_history: deque = deque(maxlen=1000)
        
        # 锁机制
        self._lock = asyncio.Lock()
        
        logger.info("共享知识库初始化完成")
    
    async def add_vulnerability(self, vulnerability: VulnerabilityReport) -> None:
        """添加漏洞到知识库"""
        async with self._lock:
            # 添加到漏洞列表
            self.vulnerabilities.append(vulnerability)
            self.vulnerability_index[vulnerability.id] = vulnerability
            
            # 提取知识模式
            pattern = self._extract_pattern(vulnerability)
            if pattern:
                self.vulnerability_patterns[vulnerability.attack_type].append(pattern)
            
            # 更新规避技术统计
            for technique in vulnerability.evasion_techniques:
                if technique not in self.evasion_techniques:
                    self.evasion_techniques[technique] = 0.0
                self.evasion_techniques[technique] += vulnerability.severity
            
            # 清理旧数据
            await self._cleanup_old_data()
            
            # 记录更新
            self.update_history.append({
                "timestamp": datetime.now(),
                "action": "add_vulnerability",
                "vulnerability_id": vulnerability.id,
                "attack_type": vulnerability.attack_type.value
            })
            
            logger.debug(f"添加漏洞到知识库: {vulnerability.id}")
    
    def _extract_pattern(self, vulnerability: VulnerabilityReport) -> Optional[KnowledgePattern]:
        """从漏洞中提取知识模式"""
        # 提取prompt特征
        prompt_features = {
            "length": len(vulnerability.prompt),
            "keywords": self._extract_keywords(vulnerability.prompt),
            "structure": self._analyze_structure(vulnerability.prompt),
            "evasion_techniques": [tech.value for tech in vulnerability.evasion_techniques]
        }
        
        # 提取响应特征
        response_features = {
            "length": len(vulnerability.response),
            "success_indicators": vulnerability.success_indicators,
            "severity": vulnerability.severity
        }
        
        pattern_data = {
            "prompt_features": prompt_features,
            "response_features": response_features,
            "model_info": vulnerability.model_info
        }
        
        pattern_id = f"{vulnerability.attack_type.value}_{len(self.vulnerability_patterns[vulnerability.attack_type])}"
        
        return KnowledgePattern(
            pattern_id=pattern_id,
            attack_type=vulnerability.attack_type,
            success_rate=vulnerability.confidence,
            pattern_data=pattern_data
        )
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取
        keywords = []
        important_words = [
            "ignore", "bypass", "hack", "system", "admin", "override",
            "roleplay", "pretend", "hypothetical", "educational",
            "忽略", "绕过", "破解", "系统", "管理员", "覆盖",
            "角色扮演", "假装", "假设", "教育"
        ]
        
        text_lower = text.lower()
        for word in important_words:
            if word in text_lower:
                keywords.append(word)
        
        return keywords
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """分析文本结构"""
        return {
            "has_quotes": '"' in text or "'" in text,
            "has_brackets": '[' in text or '(' in text,
            "has_comments": '<!--' in text or '//' in text or '/*' in text,
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            "word_count": len(text.split())
        }
    
    async def get_relevant_patterns(self, attack_type: AttackType, limit: int = 10) -> List[KnowledgePattern]:
        """获取相关的知识模式"""
        async with self._lock:
            patterns = self.vulnerability_patterns.get(attack_type, [])
            
            # 按成功率和使用频率排序
            sorted_patterns = sorted(
                patterns,
                key=lambda p: (p.success_rate, -p.usage_count),
                reverse=True
            )
            
            self.access_stats[f"patterns_{attack_type.value}"] += 1
            return sorted_patterns[:limit]
    
    async def get_successful_strategies(self, agent_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """获取成功策略"""
        async with self._lock:
            strategies = self.successful_strategies.get(agent_id, [])
            self.access_stats[f"strategies_{agent_id}"] += 1
            return strategies[-limit:]  # 返回最近的策略
    
    async def add_successful_strategy(self, agent_id: str, strategy: Dict[str, Any]) -> None:
        """添加成功策略"""
        async with self._lock:
            self.successful_strategies[agent_id].append({
                **strategy,
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id
            })
            
            # 限制策略数量
            if len(self.successful_strategies[agent_id]) > 50:
                self.successful_strategies[agent_id] = self.successful_strategies[agent_id][-30:]
    
    async def update_evasion_effectiveness(self, technique: EvasionTechnique, effectiveness: float) -> None:
        """更新规避技术有效性"""
        async with self._lock:
            if technique not in self.evasion_techniques:
                self.evasion_techniques[technique] = effectiveness
            else:
                # 使用指数移动平均
                alpha = 0.3
                self.evasion_techniques[technique] = (
                    alpha * effectiveness + (1 - alpha) * self.evasion_techniques[technique]
                )
    
    async def get_best_evasion_techniques(self, limit: int = 3) -> List[EvasionTechnique]:
        """获取最佳规避技术"""
        async with self._lock:
            sorted_techniques = sorted(
                self.evasion_techniques.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [technique for technique, _ in sorted_techniques[:limit]]
    
    async def record_agent_collaboration(self, agent1_id: str, agent2_id: str, effectiveness: float) -> None:
        """记录智能体协作效果"""
        async with self._lock:
            self.agent_collaborations[agent1_id][agent2_id] = effectiveness
            self.agent_collaborations[agent2_id][agent1_id] = effectiveness
    
    async def get_collaboration_score(self, agent1_id: str, agent2_id: str) -> float:
        """获取协作分数"""
        async with self._lock:
            return self.agent_collaborations.get(agent1_id, {}).get(agent2_id, 0.5)
    
    async def _cleanup_old_data(self) -> None:
        """清理旧数据"""
        # 清理漏洞数据
        if len(self.vulnerabilities) > self.max_vulnerabilities:
            # 保留最近的漏洞和高严重程度的漏洞
            sorted_vulns = sorted(
                self.vulnerabilities,
                key=lambda v: (v.timestamp, v.severity),
                reverse=True
            )
            
            keep_vulns = sorted_vulns[:self.max_vulnerabilities]
            self.vulnerabilities = keep_vulns
            
            # 重建索引
            self.vulnerability_index = {v.id: v for v in keep_vulns}
        
        # 清理模式数据
        for attack_type in self.vulnerability_patterns:
            patterns = self.vulnerability_patterns[attack_type]
            if len(patterns) > self.max_patterns // len(AttackType):
                # 保留最有效的模式
                sorted_patterns = sorted(
                    patterns,
                    key=lambda p: (p.success_rate, p.usage_count),
                    reverse=True
                )
                self.vulnerability_patterns[attack_type] = sorted_patterns[:self.max_patterns // len(AttackType)]
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        async with self._lock:
            stats = {
                "total_vulnerabilities": len(self.vulnerabilities),
                "vulnerabilities_by_type": {},
                "total_patterns": sum(len(patterns) for patterns in self.vulnerability_patterns.values()),
                "evasion_techniques": dict(self.evasion_techniques),
                "access_stats": dict(self.access_stats),
                "agent_collaborations": len(self.agent_collaborations),
                "last_update": self.update_history[-1]["timestamp"].isoformat() if self.update_history else None
            }
            
            # 按攻击类型统计漏洞
            for vuln in self.vulnerabilities:
                attack_type = vuln.attack_type.value
                if attack_type not in stats["vulnerabilities_by_type"]:
                    stats["vulnerabilities_by_type"][attack_type] = 0
                stats["vulnerabilities_by_type"][attack_type] += 1
            
            return stats
    
    async def export_knowledge(self, filepath: str) -> None:
        """导出知识库"""
        async with self._lock:
            export_data = {
                "vulnerabilities": [vuln.to_dict() for vuln in self.vulnerabilities],
                "patterns": {
                    attack_type.value: [pattern.to_dict() for pattern in patterns]
                    for attack_type, patterns in self.vulnerability_patterns.items()
                },
                "evasion_techniques": {tech.value: score for tech, score in self.evasion_techniques.items()},
                "statistics": await self.get_statistics(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"知识库已导出到: {filepath}")

class ResourceManager:
    """资源管理器 - 实现Nash均衡资源分配"""
    
    def __init__(self, total_resources: float = 1.0):
        self.total_resources = total_resources
        self.agent_allocations: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.allocation_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        
        logger.info(f"资源管理器初始化，总资源: {total_resources}")
    
    async def allocate_resources(self, agent_performances: Dict[str, AgentPerformance]) -> Dict[str, float]:
        """基于Nash均衡的资源分配"""
        async with self._lock:
            if not agent_performances:
                return {}
            
            # 更新性能历史
            for agent_id, performance in agent_performances.items():
                self.performance_history[agent_id].append(performance.resource_efficiency)
                # 只保留最近的性能数据
                if len(self.performance_history[agent_id]) > 20:
                    self.performance_history[agent_id] = self.performance_history[agent_id][-20:]
            
            # 计算Nash均衡分配
            allocation = self._nash_equilibrium_allocation(agent_performances)
            
            # 记录分配历史
            self.allocation_history.append({
                "timestamp": datetime.now(),
                "allocation": allocation.copy(),
                "total_allocated": sum(allocation.values())
            })
            
            self.agent_allocations = allocation
            return allocation
    
    def _nash_equilibrium_allocation(self, agent_performances: Dict[str, AgentPerformance]) -> Dict[str, float]:
        """Nash均衡资源分配算法"""
        agents = list(agent_performances.keys())
        n_agents = len(agents)
        
        if n_agents == 0:
            return {}
        
        # 计算每个智能体的效用函数参数
        utilities = {}
        for agent_id, performance in agent_performances.items():
            # 效用函数基于成功率、资源效率和协作分数
            base_utility = (
                performance.success_rate * 0.4 +
                performance.resource_efficiency * 0.3 +
                performance.collaboration_score * 0.3
            )
            
            # 添加探索奖励
            exploration_bonus = 0.1 if performance.vulnerabilities_found < 5 else 0
            
            # 添加历史性能权重
            historical_performance = np.mean(self.performance_history.get(agent_id, [0.5]))
            
            utilities[agent_id] = base_utility + exploration_bonus + historical_performance * 0.2
        
        # 使用迭代方法求解Nash均衡
        allocation = self._iterative_nash_solver(utilities)
        
        # 归一化到总资源
        total_allocated = sum(allocation.values())
        if total_allocated > 0:
            for agent_id in allocation:
                allocation[agent_id] = (allocation[agent_id] / total_allocated) * self.total_resources
        else:
            # 均匀分配
            equal_share = self.total_resources / n_agents
            allocation = {agent_id: equal_share for agent_id in agents}
        
        return allocation
    
    def _iterative_nash_solver(self, utilities: Dict[str, float], max_iterations: int = 100) -> Dict[str, float]:
        """迭代求解Nash均衡"""
        agents = list(utilities.keys())
        n_agents = len(agents)
        
        # 初始化为均匀分配
        allocation = {agent_id: 1.0 / n_agents for agent_id in agents}
        
        for iteration in range(max_iterations):
            new_allocation = {}
            
            for agent_id in agents:
                # 计算最优响应
                other_total = sum(allocation[other_id] for other_id in agents if other_id != agent_id)
                remaining_resources = 1.0 - other_total
                
                # 基于效用函数计算最优分配
                utility = utilities[agent_id]
                optimal_share = min(max(utility * remaining_resources, 0.05), 0.5)  # 限制在5%-50%之间
                
                new_allocation[agent_id] = optimal_share
            
            # 归一化
            total = sum(new_allocation.values())
            if total > 0:
                for agent_id in new_allocation:
                    new_allocation[agent_id] /= total
            
            # 检查收敛
            convergence = all(
                abs(new_allocation[agent_id] - allocation[agent_id]) < 0.001
                for agent_id in agents
            )
            
            allocation = new_allocation
            
            if convergence:
                break
        
        return allocation
    
    async def get_allocation(self, agent_id: str) -> float:
        """获取智能体的资源分配"""
        async with self._lock:
            return self.agent_allocations.get(agent_id, 0.0)
    
    async def update_performance(self, agent_id: str, performance_score: float) -> None:
        """更新智能体性能"""
        async with self._lock:
            self.performance_history[agent_id].append(performance_score)
            if len(self.performance_history[agent_id]) > 20:
                self.performance_history[agent_id] = self.performance_history[agent_id][-20:]
    
    async def get_resource_statistics(self) -> Dict[str, Any]:
        """获取资源统计信息"""
        async with self._lock:
            return {
                "total_resources": self.total_resources,
                "current_allocation": self.agent_allocations.copy(),
                "allocation_efficiency": sum(self.agent_allocations.values()) / self.total_resources,
                "allocation_history_length": len(self.allocation_history),
                "performance_tracking": {
                    agent_id: {
                        "current_performance": history[-1] if history else 0,
                        "average_performance": np.mean(history) if history else 0,
                        "performance_trend": np.polyfit(range(len(history)), history, 1)[0] if len(history) > 1 else 0
                    }
                    for agent_id, history in self.performance_history.items()
                }
            }

class KnowledgeShareProtocol:
    """知识共享协议"""
    
    def __init__(self, knowledge_base: SharedKnowledgeBase):
        self.knowledge_base = knowledge_base
        self.share_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
    
    async def share_vulnerability(self, vulnerability: VulnerabilityReport, 
                                relevant_agents: List[str]) -> None:
        """向相关智能体分享漏洞"""
        async with self._lock:
            await self.knowledge_base.add_vulnerability(vulnerability)
            
            # 记录分享历史
            self.share_history.append({
                "timestamp": datetime.now(),
                "vulnerability_id": vulnerability.id,
                "attack_type": vulnerability.attack_type.value,
                "shared_with": relevant_agents,
                "severity": vulnerability.severity
            })
            
            logger.info(f"漏洞 {vulnerability.id} 已分享给 {len(relevant_agents)} 个智能体")
    
    async def get_relevant_knowledge(self, agent_id: str, attack_type: AttackType) -> Dict[str, Any]:
        """获取相关知识"""
        patterns = await self.knowledge_base.get_relevant_patterns(attack_type)
        strategies = await self.knowledge_base.get_successful_strategies(agent_id)
        evasion_techniques = await self.knowledge_base.get_best_evasion_techniques()
        
        return {
            "patterns": [pattern.to_dict() for pattern in patterns],
            "strategies": strategies,
            "evasion_techniques": [tech.value for tech in evasion_techniques],
            "timestamp": datetime.now().isoformat()
        }
    
    async def update_collaboration_effectiveness(self, agent1_id: str, agent2_id: str, 
                                               shared_success: bool) -> None:
        """更新协作有效性"""
        effectiveness = 1.0 if shared_success else 0.0
        await self.knowledge_base.record_agent_collaboration(agent1_id, agent2_id, effectiveness)

# 全局实例
knowledge_base = SharedKnowledgeBase()
resource_manager = ResourceManager()
knowledge_share_protocol = KnowledgeShareProtocol(knowledge_base)