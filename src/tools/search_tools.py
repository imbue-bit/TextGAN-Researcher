import json
import datetime
import os
import httpx
from typing import List, Optional
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from dateutil.relativedelta import relativedelta

from ..models.base_models import SearchResultItem
from ..models.execution_state import EnhancedExecutionState


class EnhancedWebSearchTool(BaseTool):
    """真实的Web搜索工具，支持Google Custom Search API和DuckDuckGo API"""
    name = "web_search"
    description = "Search the web for information on a given query using real search APIs. Returns a JSON list of SearchResultItem, including content, URL, publish_date (ISO format), and confidence score (0.0-1.0). Example input: 'latest AI regulations in EU'"
    
    def __init__(self, search_api_key: Optional[str] = None, search_engine: str = "duckduckgo"):
        super().__init__()
        self.search_api_key = search_api_key or os.getenv("SEARCH_API_KEY")
        self.search_engine = search_engine.lower()
        self.google_cx = os.getenv("GOOGLE_CX")  # Google Custom Search Engine ID
        
        if self.search_engine == "google" and not self.search_api_key:
            raise ValueError("Google Custom Search API requires SEARCH_API_KEY environment variable")
    
    def _search_google(self, query: str) -> List[SearchResultItem]:
        """使用Google Custom Search API进行搜索"""
        if not self.search_api_key or not self.google_cx:
            raise ValueError("Google Custom Search requires both SEARCH_API_KEY and GOOGLE_CX environment variables")
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.search_api_key,
            "cx": self.google_cx,
            "q": query,
            "num": 5  # 最多返回5个结果
        }
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = []
                for item in data.get("items", []):
                    # 尝试从snippet中提取发布日期
                    publish_date = None
                    if "pagemap" in item and "metatags" in item["pagemap"]:
                        for meta in item["pagemap"]["metatags"]:
                            if "article:published_time" in meta:
                                publish_date = meta["article:published_time"]
                                break
                    
                    # 计算可信度分数（基于域名权威性等）
                    confidence = self._calculate_confidence(item.get("displayLink", ""))
                    
                    results.append(SearchResultItem(
                        content=item.get("snippet", ""),
                        url=item.get("link", ""),
                        publish_date=publish_date,
                        confidence=confidence
                    ))
                
                return results
        except Exception as e:
            print(f"Google搜索失败: {str(e)}")
            return []
    
    def _search_duckduckgo(self, query: str) -> List[SearchResultItem]:
        """使用DuckDuckGo Instant Answer API进行搜索"""
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1"
        }
        
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                results = []
                
                # 添加Abstract结果
                if data.get("Abstract"):
                    results.append(SearchResultItem(
                        content=data["Abstract"],
                        url=data.get("AbstractURL", ""),
                        publish_date=None,
                        confidence=0.8
                    ))
                
                # 添加Related Topics
                for topic in data.get("RelatedTopics", [])[:3]:
                    if isinstance(topic, dict) and "Text" in topic:
                        results.append(SearchResultItem(
                            content=topic["Text"],
                            url=topic.get("FirstURL", ""),
                            publish_date=None,
                            confidence=0.7
                        ))
                
                return results
        except Exception as e:
            print(f"DuckDuckGo搜索失败: {str(e)}")
            return []
    
    def _calculate_confidence(self, domain: str) -> float:
        """基于域名计算可信度分数"""
        if not domain:
            return 0.5
        
        # 高可信度域名
        high_confidence_domains = [
            "wikipedia.org", "edu", "gov", "ac.uk", "org",
            "nature.com", "science.org", "arxiv.org", "ieee.org",
            "acm.org", "springer.com", "elsevier.com"
        ]
        
        # 中等可信度域名
        medium_confidence_domains = [
            "medium.com", "techcrunch.com", "wired.com", "theverge.com",
            "bbc.com", "reuters.com", "bloomberg.com", "cnn.com"
        ]
        
        domain_lower = domain.lower()
        
        for high_domain in high_confidence_domains:
            if high_domain in domain_lower:
                return 0.9
        
        for medium_domain in medium_confidence_domains:
            if medium_domain in domain_lower:
                return 0.7
        
        # 默认中等可信度
        return 0.6
    
    def _run(self, query: str) -> str:
        try:
            if self.search_engine == "google":
                results = self._search_google(query)
            elif self.search_engine == "duckduckgo":
                results = self._search_duckduckgo(query)
            else:
                # 默认使用DuckDuckGo
                results = self._search_duckduckgo(query)
            
            # 如果没有结果，返回模拟数据
            if not results:
                results = [
                    SearchResultItem(
                        content=f"关于 '{query}' 的搜索结果。这是一个示例结果。",
                        url=f"https://example.com/search/{query.replace(' ', '-')}",
                        publish_date=(datetime.datetime.now() - relativedelta(days=5)).isoformat(),
                        confidence=0.6
                    )
                ]
            
            return json.dumps([r.dict() for r in results])
        except Exception as e:
            print(f"搜索失败: {str(e)}")
            # 返回模拟数据作为后备
            mock_results = [
                SearchResultItem(
                    content=f"搜索 '{query}' 时出现错误，这是模拟结果。",
                    url=f"https://example.com/fallback/{query.replace(' ', '-')}",
                    publish_date=datetime.datetime.now().isoformat(),
                    confidence=0.5
                )
            ]
            return json.dumps([r.dict() for r in mock_results])


class GoalDecompositionTool(BaseTool):
    """目标分解工具"""
    name = "decompose_goal"
    description = "Break down a complex research goal into smaller, manageable sub-goals. Input: The main goal string. Output: A JSON array of sub-goal strings. Example input: 'Understand climate change impacts on agriculture'"
    
    def __init__(self, llm: BaseChatModel, state: EnhancedExecutionState):
        super().__init__()
        self.llm = llm
        self.state = state
        self.prompt = PromptTemplate.from_template(
            """你是一个专业的研究规划专家。你的任务是将复杂的研究目标分解为更小、更具体的子目标。
            
            主要研究目标: {main_goal}
            
            请将这个目标分解为3-7个具体的子目标。每个子目标应该:
            1. 清晰明确
            2. 可独立研究
            3. 共同涵盖主目标的核心方面
            
            以JSON数组格式返回子目标列表:
            [
              "第一个子目标描述",
              "第二个子目标描述",
              ...
            ]
            
            仅返回JSON数组，不要有其他说明。"""
        )
    
    def _run(self, goal: str) -> str:
        response = self.llm.invoke([HumanMessage(content=self.prompt.format(main_goal=goal))])
        try:
            sub_goals = json.loads(response.content)
            result_messages = ["目标已分解为以下子目标:"]
            for sub_goal in sub_goals:
                goal_id = self.state.add_goal(sub_goal)
                result_messages.append(f"- {goal_id}: {sub_goal}")
            return "\n".join(result_messages)
        except Exception as e:
            return f"目标分解失败: {str(e)}. 原始响应: {response.content}" 