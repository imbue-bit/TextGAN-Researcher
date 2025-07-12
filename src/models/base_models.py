import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    """单个搜索结果的结构化表示"""
    content: str
    url: str
    publish_date: Optional[str] = None  # ISO format date string
    confidence: float = 0.7  # 0.0 to 1.0, 默认为中等可信度


class KnowledgeItem(BaseModel):
    """知识库中单个知识条目的结构化表示"""
    key: str  # 知识点的主题或名称
    value: Any  # 知识的具体内容
    source: str  # 知识来源 (e.g., URL, document name)
    publish_date: Optional[str] = None  # 知识的发布日期
    added_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())  # 加入知识库的时间
    confidence: float = 0.7  # 0.0 to 1.0, 知识的可信度
    goal_id: Optional[str] = None  # 关联的子目标ID
    verified: bool = False  # 是否经过多源验证 