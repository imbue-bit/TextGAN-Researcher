"""
Deep Research Agent API 客户端示例
演示如何使用API进行智能研究
"""

import requests
import time
import json
from typing import Dict, Any


class DeepResearchAPIClient:
    """Deep Research Agent API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def start_research(self, 
                      task_description: str,
                      openai_api_key: str,
                      search_api_key: str = None,
                      google_cx: str = None,
                      search_engine: str = "duckduckgo",
                      model_name: str = "gpt-4o-mini",
                      temperature: float = 0.5,
                      task_id: str = None) -> Dict[str, Any]:
        """启动研究任务"""
        
        payload = {
            "task_description": task_description,
            "openai_api_key": openai_api_key,
            "search_engine": search_engine,
            "model_name": model_name,
            "temperature": temperature
        }
        
        if search_api_key:
            payload["search_api_key"] = search_api_key
        if google_cx:
            payload["google_cx"] = google_cx
        if task_id:
            payload["task_id"] = task_id
        
        response = requests.post(f"{self.base_url}/research", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        response = requests.get(f"{self.base_url}/status/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def list_tasks(self) -> Dict[str, Any]:
        """列出所有任务"""
        response = requests.get(f"{self.base_url}/tasks")
        response.raise_for_status()
        return response.json()
    
    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """删除任务"""
        response = requests.delete(f"{self.base_url}/tasks/{task_id}")
        response.raise_for_status()
        return response.json()
    
    def get_config(self) -> Dict[str, Any]:
        """获取配置信息"""
        response = requests.get(f"{self.base_url}/config")
        response.raise_for_status()
        return response.json()
    
    def wait_for_completion(self, task_id: str, timeout: int = 300, check_interval: int = 5) -> Dict[str, Any]:
        """等待任务完成"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)
            
            if status["status"] in ["completed", "failed"]:
                return status
            
            print(f"任务状态: {status['status']}")
            if status.get("progress"):
                print(f"当前步骤: {status['progress'].get('current_step', '未知')}")
            
            time.sleep(check_interval)
        
        raise TimeoutError(f"任务 {task_id} 在 {timeout} 秒内未完成")


def main():
    """主函数：演示API使用"""
    
    # 创建客户端
    client = DeepResearchAPIClient()
    
    # 检查API是否运行
    try:
        config = client.get_config()
        print("API配置信息:")
        print(json.dumps(config, indent=2, ensure_ascii=False))
    except requests.exceptions.ConnectionError:
        print("错误: 无法连接到API服务器。请确保API服务器正在运行。")
        print("启动命令: python api/run_api.py")
        return
    
    # 示例：启动研究任务
    print("\n=== 启动研究任务 ===")
    
    # 注意：在实际使用中，请提供真实的API密钥
    research_request = {
        "task_description": "分析2024年全球AI芯片市场的主要趋势和竞争格局",
        "openai_api_key": "your-openai-api-key-here",  # 请替换为真实的API密钥
        "search_engine": "duckduckgo",  # 使用DuckDuckGo（免费）
        "model_name": "gpt-4o-mini",
        "temperature": 0.5
    }
    
    # 如果要使用Google搜索，需要提供以下参数：
    # "search_api_key": "your-google-api-key",
    # "google_cx": "your-custom-search-engine-id",
    # "search_engine": "google"
    
    try:
        # 启动任务
        result = client.start_research(**research_request)
        task_id = result["task_id"]
        print(f"任务已启动，ID: {task_id}")
        
        # 等待任务完成
        print("等待任务完成...")
        final_status = client.wait_for_completion(task_id, timeout=600)  # 10分钟超时
        
        if final_status["status"] == "completed":
            print("\n=== 研究完成 ===")
            print(f"任务ID: {final_status['task_id']}")
            print(f"状态: {final_status['status']}")
            print("\n研究结果:")
            print(final_status["result"])
        else:
            print(f"任务失败: {final_status.get('error', '未知错误')}")
    
    except Exception as e:
        print(f"执行失败: {str(e)}")


def simple_example():
    """简单示例：基本API调用"""
    
    client = DeepResearchAPIClient()
    
    # 1. 检查API状态
    try:
        health = requests.get(f"{client.base_url}/health").json()
        print(f"API状态: {health}")
    except:
        print("API未运行")
        return
    
    # 2. 列出所有任务
    tasks = client.list_tasks()
    print(f"当前任务数: {len(tasks['tasks'])}")
    
    # 3. 获取配置信息
    config = client.get_config()
    print("支持的模型:", config["supported_models"])
    print("支持的搜索引擎:", config["supported_search_engines"])


if __name__ == "__main__":
    # 运行完整示例
    main()
    
    # 或者运行简单示例
    # simple_example() 