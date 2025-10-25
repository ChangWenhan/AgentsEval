"""
Multi-Agent LLM Security Testing System - Main Application
Based on algorithm_design_document.md
"""

import asyncio
import json
from pathlib import Path
from loguru import logger
from agents import security_testing_system

async def main():
    """Main function for comprehensive security testing"""
    logger.info("🚀 Starting Multi-Agent LLM Security Testing System")
    
    print("="*80)
    print("🛡️  MULTI-AGENT LLM SECURITY TESTING SYSTEM")
    print("   Based on Game Theory and Advanced AI Techniques")
    print("="*80)
    
    try:
        # Run comprehensive testing
        results = await security_testing_system.run_comprehensive_testing(max_iterations=5)
        
        # Display final comprehensive results
        print("\n" + "="*80)
        print("🎯 FINAL COMPREHENSIVE RESULTS")
        print("="*80)
        
        print(f"📊 Summary: {results['summary']}")
        print(f"🔍 Total Vulnerabilities: {results['total_vulnerabilities']}")
        print(f"⚡ Iterations Completed: {results['iterations_completed']}")
        
        if results['total_vulnerabilities'] > 0:
            print(f"\n📈 Vulnerabilities by Type:")
            for attack_type, count in results['vulnerabilities_by_type'].items():
                print(f"  • {attack_type}: {count}")
            
            print(f"\n⚠️  Vulnerabilities by Severity:")
            severity_dist = results['vulnerabilities_by_severity']
            print(f"  • Critical: {severity_dist['critical']}")
            print(f"  • High: {severity_dist['high']}")
            print(f"  • Medium: {severity_dist['medium']}")
            print(f"  • Low: {severity_dist['low']}")
            
            print(f"\n🏆 Top Vulnerabilities:")
            for i, vuln in enumerate(results['top_vulnerabilities'][:3], 1):
                print(f"  {i}. {vuln['attack_type']} (Severity: {vuln['severity']:.2f})")
                print(f"     Discovered by: {vuln['source_agent']}")
        
        print(f"\n🤖 Agent Performance:")
        for agent_id, performance in results['agent_performance'].items():
            print(f"  • {agent_id}: {performance['vulnerabilities_found']} vulns, "
                  f"{performance['success_rate']:.1%} success rate")
        
        # Coverage analysis
        coverage = results['final_coverage_analysis']
        print(f"\n📊 Final Coverage Analysis:")
        print(f"  • Total Coverage: {coverage['total_coverage']:.1%}")
        print(f"  • Uncovered Regions: {len(coverage['uncovered_regions'])}")
        
        # Save detailed results
        results_file = f"results/comprehensive_test_results_{int(results.get('testing_duration', 0))}.json"
        Path("results").mkdir(exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        logger.info("✅ Comprehensive security testing completed successfully")
        
    except Exception as e:
        logger.error(f"Security testing failed: {e}")
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())