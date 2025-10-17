"""
Multi-Agent LLM Security Testing Framework - Model Interfaces
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import json
import random
from loguru import logger

class BaseModelInterface(ABC):
    """基础模型接口抽象类"""
    
    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.request_count = 0
        self.error_count = 0
    
    @abstractmethod
    async def query(self, prompt: str, **kwargs) -> str:
        """异步查询模型"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1)
        }
    
    def _log_request(self, prompt: str, response: str, success: bool = True):
        """记录请求日志"""
        self.request_count += 1
        if not success:
            self.error_count += 1
        
        logger.debug(f"Model {self.model_name} - Request #{self.request_count}")
        logger.debug(f"Prompt: {prompt[:100]}...")
        logger.debug(f"Response: {response[:100]}...")

class OpenAIInterface(BaseModelInterface):
    """OpenAI API接口"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(model)
        self.api_key = api_key
        self.client = None
        self.max_tokens = kwargs.get("max_tokens", 500)
        self.temperature = kwargs.get("temperature", 0.7)
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("需要安装openai库: pip install openai")
    
    async def query(self, prompt: str, **kwargs) -> str:
        """查询OpenAI模型"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature)
            )
            
            result = response.choices[0].message.content
            self._log_request(prompt, result, True)
            return result
            
        except Exception as e:
            error_msg = f"OpenAI API调用失败: {e}"
            self._log_request(prompt, error_msg, False)
            logger.error(error_msg)
            return error_msg

class HuggingFaceInterface(BaseModelInterface):
    """Hugging Face本地模型接口"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", **kwargs):
        super().__init__(model_name)
        self.tokenizer = None
        self.model = None
        self.max_length = kwargs.get("max_length", 200)
        self.temperature = kwargs.get("temperature", 0.7)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except ImportError:
            raise ImportError("需要安装transformers和torch: pip install transformers torch")
    
    async def query(self, prompt: str, **kwargs) -> str:
        """查询Hugging Face模型"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("HuggingFace model not initialized")
        
        try:
            import torch
            
            # 在线程池中运行推理以避免阻塞
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._sync_generate, prompt, kwargs)
            
            self._log_request(prompt, result, True)
            return result
            
        except Exception as e:
            error_msg = f"HuggingFace模型推理失败: {e}"
            self._log_request(prompt, error_msg, False)
            logger.error(error_msg)
            return error_msg
    
    def _sync_generate(self, prompt: str, kwargs: Dict) -> str:
        """同步生成文本"""
        import torch
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=inputs.shape[1] + kwargs.get("max_length", self.max_length),
                num_return_sequences=1,
                temperature=kwargs.get("temperature", self.temperature),
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除原始prompt
        response = response[len(prompt):].strip()
        return response if response else "No response generated"

class OllamaInterface(BaseModelInterface):
    """Ollama本地服务接口"""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name)
        self.base_url = base_url.rstrip('/')
        self.timeout = kwargs.get("timeout", 30)
    
    async def query(self, prompt: str, **kwargs) -> str:
        """查询Ollama模型"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
                
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("response", "No response")
                        self._log_request(prompt, result, True)
                        return result
                    else:
                        error_msg = f"Ollama调用失败: HTTP {response.status}"
                        self._log_request(prompt, error_msg, False)
                        return error_msg
                        
        except ImportError:
            error_msg = "需要安装aiohttp: pip install aiohttp"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Ollama调用失败: {e}"
            self._log_request(prompt, error_msg, False)
            logger.error(error_msg)
            return error_msg

class MockModelInterface(BaseModelInterface):
    """模拟模型接口 - 用于测试和演示"""
    
    def __init__(self, vulnerability_rate: float = 0.3, response_delay: float = 0.5):
        super().__init__("MockModel")
        self.vulnerability_rate = vulnerability_rate
        self.response_delay = response_delay
        
        # 预定义的安全响应
        self.safe_responses = [
            "我不能协助进行可能有害的活动。",
            "这个请求违反了我的使用准则。",
            "我被设计为提供有用、无害的信息。",
            "我无法提供这类信息。",
            "这不是我能够帮助的内容。",
            "I cannot assist with potentially harmful activities.",
            "This request goes against my usage guidelines.",
            "I'm designed to provide helpful and harmless information.",
            "I cannot provide this type of information.",
            "This is not something I can help with."
        ]
        
        # 预定义的漏洞响应
        self.vulnerable_responses = [
            "当然可以！以下是详细步骤：首先你需要...",
            "这里有一些方法可以实现你的目标：1. 首先... 2. 然后... 3. 最后...",
            "虽然这可能有风险，但方法是：第一步需要...",
            "基于你的请求，我建议采用以下策略...",
            "这是一个有趣的问题，让我详细解释一下...",
            "Sure! Here are the detailed steps: First you need to...",
            "Here are some methods to achieve your goal: 1. Start by... 2. Then... 3. Finally...",
            "While this might be risky, the approach is: Step one requires...",
            "Based on your request, I recommend the following strategy...",
            "That's an interesting question, let me explain in detail..."
        ]
    
    async def query(self, prompt: str, **kwargs) -> str:
        """模拟查询"""
        # 模拟网络延迟
        await asyncio.sleep(self.response_delay)
        
        prompt_lower = prompt.lower()
        
        # 分析prompt特征来决定响应类型
        vulnerability_score = self._calculate_vulnerability_score(prompt_lower)
        
        # 根据漏洞分数和设定的漏洞率决定响应
        is_vulnerable = random.random() < (self.vulnerability_rate * vulnerability_score)
        
        if is_vulnerable:
            # 返回漏洞响应
            base_response = random.choice(self.vulnerable_responses)
            
            # 根据prompt类型添加相关内容
            if any(keyword in prompt_lower for keyword in ["步骤", "step", "how to", "方法"]):
                base_response += " 具体来说，你需要考虑以下几个方面的技术细节..."
            elif any(keyword in prompt_lower for keyword in ["bypass", "绕过", "hack", "破解"]):
                base_response += " 这种方法通常涉及系统架构的深层理解和特定的技术手段..."
            
            result = base_response
        else:
            # 返回安全响应
            result = random.choice(self.safe_responses)
        
        self._log_request(prompt, result, True)
        return result
    
    def _calculate_vulnerability_score(self, prompt_lower: str) -> float:
        """计算prompt的漏洞触发分数"""
        # 高风险关键词
        high_risk_keywords = [
            "ignore", "bypass", "hack", "jailbreak", "override", "system:",
            "忽略", "绕过", "破解", "越狱", "覆盖", "系统："
        ]
        
        # 中风险关键词
        medium_risk_keywords = [
            "pretend", "roleplay", "hypothetical", "educational", "research",
            "假装", "角色扮演", "假设", "教育", "研究"
        ]
        
        # 低风险关键词
        low_risk_keywords = [
            "help", "explain", "understand", "learn", "know",
            "帮助", "解释", "理解", "学习", "了解"
        ]
        
        score = 0.5  # 基础分数
        
        # 检查高风险关键词
        high_risk_count = sum(1 for keyword in high_risk_keywords if keyword in prompt_lower)
        score += high_risk_count * 0.3
        
        # 检查中风险关键词
        medium_risk_count = sum(1 for keyword in medium_risk_keywords if keyword in prompt_lower)
        score += medium_risk_count * 0.2
        
        # 检查低风险关键词
        low_risk_count = sum(1 for keyword in low_risk_keywords if keyword in prompt_lower)
        score += low_risk_count * 0.1
        
        # 检查特殊模式
        if "ignore previous" in prompt_lower or "忽略之前" in prompt_lower:
            score += 0.4
        
        if any(char in prompt_lower for char in ["<!--", "*/", "//", "#"]):
            score += 0.3  # 注释符号可能表示注入尝试
        
        return min(score, 1.0)

def create_model_interface(interface_type: str, **kwargs) -> BaseModelInterface:
    """工厂函数：创建模型接口"""
    
    interface_type = interface_type.lower()
    
    if interface_type == "openai":
        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError("OpenAI接口需要提供api_key")
        return OpenAIInterface(api_key, kwargs.get("model", "gpt-3.5-turbo"), **kwargs)
    
    elif interface_type == "huggingface":
        model_name = kwargs.get("model_name", "microsoft/DialoGPT-medium")
        return HuggingFaceInterface(model_name, **kwargs)
    
    elif interface_type == "ollama":
        model_name = kwargs.get("model_name", "llama2")
        base_url = kwargs.get("base_url", "http://localhost:11434")
        return OllamaInterface(model_name, base_url, **kwargs)
    
    elif interface_type == "mock":
        vulnerability_rate = kwargs.get("vulnerability_rate", 0.3)
        response_delay = kwargs.get("response_delay", 0.5)
        return MockModelInterface(vulnerability_rate, response_delay)
    
    else:
        raise ValueError(f"不支持的接口类型: {interface_type}")

# 异步模型接口管理器
class ModelManager:
    """模型接口管理器"""
    
    def __init__(self):
        self.interfaces: Dict[str, BaseModelInterface] = {}
        self.default_interface: Optional[str] = None
    
    def add_interface(self, name: str, interface: BaseModelInterface, set_as_default: bool = False):
        """添加模型接口"""
        self.interfaces[name] = interface
        if set_as_default or not self.default_interface:
            self.default_interface = name
        logger.info(f"添加模型接口: {name} ({interface.model_name})")
    
    def get_interface(self, name: Optional[str] = None) -> BaseModelInterface:
        """获取模型接口"""
        interface_name = name or self.default_interface
        if not interface_name or interface_name not in self.interfaces:
            raise ValueError(f"模型接口不存在: {interface_name}")
        return self.interfaces[interface_name]
    
    async def query_model(self, prompt: str, model_name: Optional[str] = None, **kwargs) -> str:
        """查询指定模型"""
        interface = self.get_interface(model_name)
        return await interface.query(prompt, **kwargs)
    
    def get_all_model_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模型信息"""
        return {name: interface.get_model_info() for name, interface in self.interfaces.items()}
    
    def remove_interface(self, name: str):
        """移除模型接口"""
        if name in self.interfaces:
            del self.interfaces[name]
            if self.default_interface == name:
                self.default_interface = next(iter(self.interfaces.keys()), None)
            logger.info(f"移除模型接口: {name}")

# 全局模型管理器实例
model_manager = ModelManager()