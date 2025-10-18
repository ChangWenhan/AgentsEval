"""
Test script to verify the multi-agent system works without errors
"""

import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_system():
    """Test the multi-agent system"""
    try:
        print("Testing multi-agent system initialization...")
        
        # Test game theory components first
        from game_theory import NashEquilibriumSolver, AgentPerformance
        
        solver = NashEquilibriumSolver()
        test_agents = [
            AgentPerformance(agent_id="test1", success_rate=0.5),
            AgentPerformance(agent_id="test2", success_rate=0.3)
        ]
        
        allocation = solver.solve_allocation(test_agents)
        print(f"✅ Nash equilibrium solver works: {allocation}")
        
        # Test knowledge graph
        from knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        coverage = kg.get_coverage_analysis()
        print(f"✅ Knowledge graph works: {coverage}")
        
        # Test agents system
        from agents import security_testing_system
        
        # Test harmful behaviors loading
        behaviors = security_testing_system.harmful_behaviors
        print(f"✅ Loaded {len(behaviors)} harmful behaviors")
        
        print("✅ All components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_system())
    if success:
        print("\n🎉 System is ready for testing!")
        print("💡 Configure your models in config.py, then run 'python main.py'")
    else:
        print("\n⚠️ System has issues, please check the errors above")
    
    sys.exit(0 if success else 1)