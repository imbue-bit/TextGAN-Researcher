"""
辅助函数模块
包含一些通用的工具函数
"""

import datetime
from typing import List, Dict, Any
from dateutil.parser import parse as parse_date


def format_timestamp(timestamp: str) -> str:
    """格式化时间戳为可读格式"""
    try:
        dt = parse_date(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp


def truncate_text(text: str, max_length: int = 100) -> str:
    """截断文本到指定长度"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def calculate_confidence_score(sources: List[Dict[str, Any]]) -> float:
    """根据多个来源计算综合可信度分数"""
    if not sources:
        return 0.0
    
    total_confidence = sum(source.get('confidence', 0.5) for source in sources)
    return total_confidence / len(sources)


def get_recent_items(items: List[Dict[str, Any]], days: int = 30) -> List[Dict[str, Any]]:
    """获取指定天数内的项目"""
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    recent_items = []
    
    for item in items:
        try:
            if 'timestamp' in item:
                item_date = parse_date(item['timestamp'])
                if item_date >= cutoff_date:
                    recent_items.append(item)
        except:
            continue
    
    return recent_items 