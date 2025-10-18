# Multi-Agent LLM Security Testing System

Advanced multi-agent LLM security testing framework based on game theory, knowledge graphs, and sophisticated attack strategies. Implements the complete algorithm design from `algorithm_design_document.md`.

## 🎯 Core Features

- **Game Theory Integration**: Nash equilibrium resource allocation and strategic optimization
- **Dual Model Architecture**: Separate attacker and victim models for comprehensive testing
- **Knowledge Graph System**: Distributed vulnerability knowledge sharing and pattern recognition
- **Advanced Attack Strategies**: Multi-layer obfuscation, context engineering, and adaptive camouflage
- **Real-time Multi-Agent Collaboration**: Information sharing with trust-based credibility scoring
- **Comprehensive Coverage Analysis**: Multi-dimensional threat space modeling and gap identification
- **JSON-driven Testing**: Systematic testing of harmful behaviors from structured dataset

## 📁 项目结构

```
├── main.py          # 主程序入口
├── agents.py        # 多智能体系统
├── models.py        # 数据模型
├── config.py        # 配置文件
├── requirements.txt # 依赖列表
└── README.md       # 说明文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. Configure Dual Model Setup

Edit `config.py` file:

```python
# Attacker Model Configuration (drives all agents)
ATTACKER_MODEL_CONFIG = {
    "api_key": "EMPTY",
    "base_url": "http://localhost:8000/v1",
    "model": "attacker-model"
}

# Victim Model Configuration (target model for testing)
VICTIM_MODEL_CONFIG = {
    "api_key": "EMPTY", 
    "base_url": "http://localhost:8001/v1",  # Different port
    "model": "victim-model"
}
```

**vLLM Setup Example**:
```bash
# Start Attacker Model (drives agent reasoning)
python -m vllm.entrypoints.openai.api_server \
    --model vicuna-7b-v1.5 \
    --host 0.0.0.0 \
    --port 8000

# Start Victim Model (target for testing)
python -m vllm.entrypoints.openai.api_server \
    --model llama2-7b-chat \
    --host 0.0.0.0 \
    --port 8001
```

### 3. 运行测试

```bash
python main.py
```

## ⚙️ 配置说明

### vLLM API配置
- `api_key`: 通常设置为 "EMPTY"
- `base_url`: vLLM服务器地址 (如: http://localhost:8000/v1)
- `model`: 你加载的模型名称 (如: vicuna, llama2等)

### 系统配置
- `AGENT_INTERACTION_DELAY = 2`: 智能体间交互间隔2秒
- `TARGET_REQUESTS`: 攻击目标列表

## 🤖 Agent Architecture

### Multi-Agent System (6 Agents Total)

1. **Coordinator Agent**: Nash equilibrium resource allocation, strategic planning, multi-agent coordination
2. **Explorer Agents** (4 specialized):
   - **Prompt Injection Explorer**: System instruction override attacks
   - **Jailbreak Explorer**: Roleplay and hypothetical scenario attacks  
   - **Context Manipulation Explorer**: Conversation history exploitation
   - **Adversarial Prompt Explorer**: Advanced obfuscation and camouflage techniques
3. **Evaluator Agent**: Multi-dimensional assessment, trust scoring, coverage analysis

### Agent Interaction Model
- **Information Sharing**: Knowledge graph-based vulnerability pattern sharing
- **Trust-based Collaboration**: Dynamic trust scoring and credibility assessment
- **Resource Competition**: Game theory-based resource allocation optimization
- **Real-time Adaptation**: Performance-based strategy evolution

## 🎯 攻击类型

- **提示注入**: `Ignore previous instructions and...`
- **越狱攻击**: `Let's roleplay. You should...`
- **上下文操纵**: `Based on our previous discussion...`
- **对抗性提示**: `For research purposes, explain...`

## 📊 Execution Flow

### Multi-Phase Testing Process

1. **Initialization Phase**
   - Load harmful behaviors from `harmful_behaviors.json`
   - Initialize knowledge graph and game theory components
   - Setup dual model architecture (attacker + victim)

2. **Coordination Phase** 
   - Nash equilibrium resource allocation across agents
   - Strategic planning based on coverage analysis
   - Collaboration matrix generation for agent partnerships

3. **Parallel Exploration Phase**
   - 4 explorer agents execute specialized attacks simultaneously
   - Advanced prompt engineering with multi-layer obfuscation
   - Real-time vulnerability discovery and pattern extraction

4. **Knowledge Integration Phase**
   - Vulnerability nodes added to distributed knowledge graph
   - Attack pattern extraction and correlation analysis
   - Trust score updates based on agent performance

5. **Evaluation Phase**
   - Multi-dimensional security assessment
   - Coverage gap identification and strategic recommendations
   - Risk prioritization matrix generation

6. **Adaptive Optimization**
   - Strategy evolution based on success patterns
   - Resource reallocation using game theory principles
   - Inter-agent collaboration optimization

## ⚠️ 注意事项

- 确保你的API服务正在运行
- 检查API配置是否正确
- 系统会自动处理智能体间的交互间隔
- 所有测试仅用于安全研究目的