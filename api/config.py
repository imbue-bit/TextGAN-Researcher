"""
Deep Research Agent API 配置文件
"""

import os
from typing import List


class APIConfig:
    """API配置类"""
    
    # 服务器配置
    HOST = os.getenv("API_HOST", "0.0.0.0")
    PORT = int(os.getenv("API_PORT", "8000"))
    RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
    
    # 应用配置
    TITLE = "Deep Research Agent API"
    DESCRIPTION = "基于TextGAN-D架构的智能研究代理API"
    VERSION = "1.0.0"
    
    # 支持的模型
    SUPPORTED_MODELS = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo"
    ]
    
    # 支持的搜索引擎
    SUPPORTED_SEARCH_ENGINES = [
        "duckduckgo",
        "google"
    ]
    
    # 必需的环境变量
    REQUIRED_ENV_VARS = [
        "OPENAI_API_KEY"
    ]
    
    # 可选的环境变量
    OPTIONAL_ENV_VARS = [
        "SEARCH_API_KEY",
        "GOOGLE_CX"
    ]
    
    # 任务超时配置
    DEFAULT_TASK_TIMEOUT = 600  # 10分钟
    MAX_TASK_TIMEOUT = 3600     # 1小时
    
    # 任务存储配置
    MAX_TASKS_IN_MEMORY = 100   # 内存中最大任务数
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """验证配置并返回错误列表"""
        errors = []
        
        # 检查必需的环境变量
        for var in cls.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                errors.append(f"缺少必需的环境变量: {var}")
        
        # 检查端口范围
        if not (1024 <= cls.PORT <= 65535):
            errors.append(f"端口号必须在1024-65535之间: {cls.PORT}")
        
        return errors
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """获取配置摘要"""
        return {
            "server": {
                "host": cls.HOST,
                "port": cls.PORT,
                "reload": cls.RELOAD
            },
            "app": {
                "title": cls.TITLE,
                "version": cls.VERSION
            },
            "supported_models": cls.SUPPORTED_MODELS,
            "supported_search_engines": cls.SUPPORTED_SEARCH_ENGINES,
            "required_env_vars": cls.REQUIRED_ENV_VARS,
            "optional_env_vars": cls.OPTIONAL_ENV_VARS,
            "task_config": {
                "default_timeout": cls.DEFAULT_TASK_TIMEOUT,
                "max_timeout": cls.MAX_TASK_TIMEOUT,
                "max_tasks_in_memory": cls.MAX_TASKS_IN_MEMORY
            }
        } 