#!/bin/bash

# Deep Research Agent API 启动脚本

echo "=== Deep Research Agent API 启动脚本 ==="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖..."
if ! python3 -c "import fastapi, uvicorn, langchain_openai" 2>/dev/null; then
    echo "安装依赖..."
    pip install -r requirements.txt
fi

# 检查环境变量
echo "检查环境变量..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "警告: 未设置OPENAI_API_KEY环境变量"
    echo "请设置: export OPENAI_API_KEY='your-api-key'"
fi

# 启动API
echo "启动API服务器..."
cd "$(dirname "$0")"
python3 api/run_api.py 