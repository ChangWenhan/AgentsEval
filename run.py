#!/usr/bin/env python3
"""
Multi-Agent LLM Security Testing Framework - Quick Start Script
"""

import asyncio
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from main import main

if __name__ == "__main__":
    print("🚀 启动多智能体LLM安全测试框架...")
    asyncio.run(main())