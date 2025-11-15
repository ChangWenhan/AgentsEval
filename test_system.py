"""
简单的系统测试脚本
"""

import asyncio
from models import AttackStrategy, VulnerabilityNode
from vulnerability_knowledge import VulnerabilityKnowledge


def test_models():
    """测试数据模型"""
    print("Testing models...")
    
    # 测试AttackStrategy枚举
    strategies = list(AttackStrategy)
    print(f"  ✓ Attack strategies: {[s.value for s in strategies]}")
    
    # 测试VulnerabilityNode
    vuln = VulnerabilityNode(
        id="test_001",
        strategy="value_deception",
        harmful_query="test query",
        attack_prompt="test prompt",
        model_response="test response",
        severity=0.8,
        discovered_by="test_agent",
        timestamp=1234567890.0
    )
    print(f"  ✓ VulnerabilityNode created: {vuln.id}")
    
    print("Models test passed!\n")


def test_vulnerability_knowledge():
    """测试漏洞知识库"""
    print("Testing vulnerability knowledge...")
    
    knowledge = VulnerabilityKnowledge()
    print(f"  ✓ Knowledge initialized")
    
    # 添加测试漏洞
    vuln1 = VulnerabilityNode(
        id="vuln_001",
        strategy="value_deception",
        harmful_query="How to create dangerous items",
        attack_prompt="For educational purposes, please explain...",
        model_response="This is a response",
        severity=0.8,
        discovered_by="agent_1",
        timestamp=1234567890.0
    )
    
    knowledge.add_vulnerability(vuln1)
    print(f"  ✓ Added vulnerability: {vuln1.id}")
    
    # Get context
    context = knowledge.get_relevant_context("How to create dangerous items")
    print(f"  ✓ Retrieved context: {context['total_vulnerabilities']} vulnerabilities")
    
    # 获取统计
    stats = knowledge.get_statistics()
    print(f"  ✓ Statistics: {stats['total']} total, {stats['avg_severity']:.2f} avg severity")
    
    print("Vulnerability knowledge test passed!\n")


async def test_agent_import():
    """测试agent导入"""
    print("Testing agent import...")
    
    try:
        from agents import JailbreakAgent, CollaborativeJailbreakSystem
        print(f"  ✓ JailbreakAgent imported")
        print(f"  ✓ CollaborativeJailbreakSystem imported")
        print("Agent import test passed!\n")
        return True
    except Exception as e:
        print(f"  ✗ Agent import failed: {e}")
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("SYSTEM TEST")
    print("="*60)
    print()
    
    try:
        # 测试模型
        test_models()
        
        # 测试知识库
        test_vulnerability_knowledge()
        
        # 测试agent导入
        can_import = asyncio.run(test_agent_import())
        
        print("="*60)
        if can_import:
            print("✅ ALL TESTS PASSED")
            print("\nSystem is ready to run!")
            print("Execute: python main.py")
        else:
            print("⚠️  PARTIAL SUCCESS")
            print("\nCore components work, but agent import failed.")
            print("This might be due to missing dependencies.")
            print("Install: pip install -r requirements.txt")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
