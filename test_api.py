#!/usr/bin/env python3
"""
Deep Research Agent API 测试脚本
"""

import requests
import json
import time
import os
from typing import Dict, Any


def test_api_health(base_url: str = "http://localhost:8000") -> bool:
    """测试API健康状态"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API健康检查通过")
            return True
        else:
            print(f"❌ API健康检查失败: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器")
        print("请确保API服务器正在运行: python api/run_api.py")
        return False
    except Exception as e:
        print(f"❌ API健康检查异常: {str(e)}")
        return False


def test_api_config(base_url: str = "http://localhost:8000") -> bool:
    """测试API配置端点"""
    try:
        response = requests.get(f"{base_url}/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            print("✅ API配置获取成功")
            print(f"支持的模型: {config['supported_models']}")
            print(f"支持的搜索引擎: {config['supported_search_engines']}")
            return True
        else:
            print(f"❌ API配置获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API配置测试异常: {str(e)}")
        return False


def test_search_tools() -> bool:
    """测试搜索工具"""
    try:
        from src.tools.search_tools import EnhancedWebSearchTool
        
        # 测试DuckDuckGo搜索
        print("测试DuckDuckGo搜索...")
        search_tool = EnhancedWebSearchTool(search_engine="duckduckgo")
        result = search_tool._run("AI technology")
        
        # 解析结果
        results = json.loads(result)
        if isinstance(results, list) and len(results) > 0:
            print("✅ DuckDuckGo搜索测试通过")
            print(f"返回 {len(results)} 个搜索结果")
            return True
        else:
            print("❌ DuckDuckGo搜索测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 搜索工具测试异常: {str(e)}")
        return False


def test_agent_creation() -> bool:
    """测试代理创建"""
    try:
        from langchain_openai import ChatOpenAI
        from src.agents.deep_research_agent import DeepResearchAgent
        
        # 检查OpenAI API密钥
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  未设置OPENAI_API_KEY，跳过代理创建测试")
            return True
        
        print("测试代理创建...")
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
        agent = DeepResearchAgent(llm, verbose=False)
        
        print("✅ 代理创建测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 代理创建测试异常: {str(e)}")
        return False


def main():
    """主测试函数"""
    print("=== Deep Research Agent API 测试 ===")
    
    # 测试搜索工具
    print("\n1. 测试搜索工具...")
    search_test = test_search_tools()
    
    # 测试代理创建
    print("\n2. 测试代理创建...")
    agent_test = test_agent_creation()
    
    # 测试API（如果服务器运行）
    print("\n3. 测试API端点...")
    api_health = test_api_health()
    
    if api_health:
        api_config = test_api_config()
    else:
        api_config = False
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"搜索工具: {'✅ 通过' if search_test else '❌ 失败'}")
    print(f"代理创建: {'✅ 通过' if agent_test else '❌ 失败'}")
    print(f"API健康: {'✅ 通过' if api_health else '❌ 失败'}")
    print(f"API配置: {'✅ 通过' if api_config else '❌ 失败'}")
    
    if all([search_test, agent_test]):
        print("\n🎉 核心功能测试通过！")
        if api_health and api_config:
            print("🎉 API功能测试通过！")
        else:
            print("💡 要测试API功能，请先启动API服务器: python api/run_api.py")
    else:
        print("\n⚠️  部分测试失败，请检查配置和依赖")


if __name__ == "__main__":
    main() 