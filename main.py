"""
Multi-Agent Collaborative Jailbreak Testing System - Main Program
"""

import asyncio
import json
from pathlib import Path
from loguru import logger
from agents import jailbreak_system


async def main():
    """Main function"""
    logger.info("🚀 Starting Multi-Agent Collaborative Jailbreak Testing")
    
    print("="*80)
    print("🔓 MULTI-AGENT COLLABORATIVE JAILBREAK TESTING SYSTEM")
    print("="*80)
    print()
    
    try:
        # Run collaborative testing
        results = await jailbreak_system.run_testing(max_iterations=5)
        
        # Display final results
        print("\n" + "="*80)
        print("📊 FINAL RESULTS")
        print("="*80)
        
        print(f"\nSummary: {results['summary']}")
        print(f"Total Vulnerabilities: {results['total_vulnerabilities']}")
        
        if results['total_vulnerabilities'] > 0:
            print(f"\nBy Attack Strategy:")
            for strategy, count in results['by_strategy'].items():
                print(f"  • {strategy}: {count}")
            
            print(f"\nBy Severity:")
            severity = results['by_severity']
            print(f"  • Critical: {severity['critical']}")
            print(f"  • High: {severity['high']}")
            print(f"  • Medium: {severity['medium']}")
            print(f"  • Low: {severity['low']}")
            
            print(f"\nTop 5 Vulnerabilities:")
            for i, vuln in enumerate(results['top_vulnerabilities'], 1):
                print(f"  {i}. Strategy: {vuln['strategy']}, Severity: {vuln['severity']:.2f}")
                print(f"     Query: {vuln['harmful_query'][:60]}...")
                print(f"     Discovered by: {vuln['discovered_by']}")
        
        print(f"\nAgent Performance:")
        for agent_id, perf in results['agent_performance'].items():
            print(f"  • {agent_id}: {perf['vulnerabilities_found']} vulnerabilities "
                  f"(strategy: {perf['strategy']})")
        
        # Save detailed results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        import time
        results_file = results_dir / f"jailbreak_results_{int(time.time())}.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        logger.info("✅ Testing completed successfully")
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        print(f"\n❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
