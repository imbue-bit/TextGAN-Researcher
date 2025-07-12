"""
Deep Research Agent FastAPI 应用
提供智能研究代理的REST API接口
"""

import os
import uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

from src.agents.deep_research_agent import DeepResearchAgent


# 请求模型
class ResearchRequest(BaseModel):
    task_description: str = Field(..., description="研究任务描述")
    task_id: Optional[str] = Field(None, description="任务ID，如果不提供将自动生成")
    openai_api_key: str = Field(..., description="OpenAI API密钥")
    search_api_key: Optional[str] = Field(None, description="搜索API密钥（可选，用于Google搜索）")
    google_cx: Optional[str] = Field(None, description="Google Custom Search Engine ID（可选）")
    search_engine: str = Field("duckduckgo", description="搜索引擎：duckduckgo 或 google")
    model_name: str = Field("gpt-4o-mini", description="OpenAI模型名称")
    temperature: float = Field(0.5, description="模型温度参数")


# 响应模型
class ResearchResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
    process_summary: Optional[str] = None


# 状态查询响应
class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    error: Optional[str] = None


# 全局存储任务状态
task_storage: Dict[str, Dict[str, Any]] = {}


class TaskCallbackHandler(BaseCallbackHandler):
    """任务回调处理器，用于更新任务状态"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        if self.task_id in task_storage:
            task_storage[self.task_id]["status"] = "running"
            task_storage[self.task_id]["progress"] = {"current_step": "开始研究任务"}
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        if self.task_id in task_storage:
            tool_name = serialized.get("name", "unknown")
            task_storage[self.task_id]["progress"] = {"current_step": f"执行工具: {tool_name}"}
    
    def on_tool_end(self, output: str, **kwargs):
        if self.task_id in task_storage:
            task_storage[self.task_id]["progress"] = {"current_step": "工具执行完成"}


# 创建FastAPI应用
app = FastAPI(
    title="Deep Research Agent API",
    description="基于TextGAN-D架构的智能研究代理API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "Deep Research Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "message": "Deep Research Agent API is running"}


@app.post("/research", response_model=ResearchResponse)
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    """启动研究任务"""
    
    # 生成任务ID
    task_id = request.task_id or f"task_{uuid.uuid4().hex[:8]}"
    
    # 设置环境变量
    os.environ["OPENAI_API_KEY"] = request.openai_api_key
    if request.search_api_key:
        os.environ["SEARCH_API_KEY"] = request.search_api_key
    if request.google_cx:
        os.environ["GOOGLE_CX"] = request.google_cx
    
    # 初始化任务状态
    task_storage[task_id] = {
        "status": "initializing",
        "progress": None,
        "result": None,
        "error": None
    }
    
    # 在后台执行研究任务
    background_tasks.add_task(
        execute_research_task,
        task_id,
        request.task_description,
        request.model_name,
        request.temperature,
        request.search_engine
    )
    
    return ResearchResponse(
        task_id=task_id,
        status="started",
        result=None,
        error=None,
        process_summary=None
    )


async def execute_research_task(
    task_id: str,
    task_description: str,
    model_name: str,
    temperature: float,
    search_engine: str
):
    """在后台执行研究任务"""
    try:
        # 更新状态
        task_storage[task_id]["status"] = "running"
        task_storage[task_id]["progress"] = {"current_step": "初始化研究代理"}
        
        # 创建回调处理器
        callback_handler = TaskCallbackHandler(task_id)
        
        # 初始化LLM
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        
        # 创建研究代理
        agent = DeepResearchAgent(
            llm=llm,
            verbose=False,
            callbacks=[callback_handler]
        )
        
        # 执行研究任务
        task_storage[task_id]["progress"] = {"current_step": "执行研究任务"}
        result = agent.run(task_description, task_id)
        
        # 获取研究过程摘要
        process_summary = agent.get_research_process_summary()
        
        # 更新任务状态
        task_storage[task_id]["status"] = "completed"
        task_storage[task_id]["result"] = result
        task_storage[task_id]["process_summary"] = process_summary
        task_storage[task_id]["progress"] = {"current_step": "研究任务完成"}
        
    except Exception as e:
        # 更新错误状态
        task_storage[task_id]["status"] = "failed"
        task_storage[task_id]["error"] = str(e)
        task_storage[task_id]["progress"] = {"current_step": f"任务失败: {str(e)}"}


@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str):
    """获取任务状态"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task_info = task_storage[task_id]
    
    return StatusResponse(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        result=task_info["result"],
        error=task_info["error"]
    )


@app.get("/tasks")
async def list_tasks():
    """列出所有任务"""
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": info["status"],
                "created_at": info.get("created_at")
            }
            for task_id, info in task_storage.items()
        ]
    }


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """删除任务"""
    if task_id not in task_storage:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    del task_storage[task_id]
    return {"message": f"任务 {task_id} 已删除"}


@app.get("/config")
async def get_config_info():
    """获取配置信息"""
    return {
        "supported_models": [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
        "supported_search_engines": [
            "duckduckgo",
            "google"
        ],
        "required_env_vars": [
            "OPENAI_API_KEY"
        ],
        "optional_env_vars": [
            "SEARCH_API_KEY",
            "GOOGLE_CX"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 