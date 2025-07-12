import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

from .base_models import SearchResultItem, KnowledgeItem


class EnhancedExecutionState(BaseModel):
    """增强的执行状态，支持目标分解、资料时效性管理、假设与矛盾记录"""
    
    task_id: str
    task_description: str
    decomposed_goals: List[Dict[str, Any]] = Field(default_factory=list)  # {id, description, status, created_at, completed_at, related_searches, related_knowledge}
    search_iterations: List[Dict[str, Any]] = Field(default_factory=list)  # {id, timestamp, query, results: List[SearchResultItem], goal_id}
    knowledge_base: List[KnowledgeItem] = Field(default_factory=list)  # List of KnowledgeItem
    hypotheses: List[Dict[str, Any]] = Field(default_factory=list)  # {id, hypothesis, status (pending_verification, supported, refuted), supporting_knowledge_ids, contradicting_knowledge_ids}
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)  # {id, description, conflicting_knowledge_ids, resolution_attempt_description, status (unresolved, resolving, resolved)}
    
    current_generation: Optional[str] = None
    current_score: Optional[float] = None
    current_criticism: Optional[str] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)  # 历史生成记录
    
    def add_goal(self, goal: str, status: str = "pending") -> str:
        """添加一个分解后的子目标"""
        goal_id = f"goal_{len(self.decomposed_goals) + 1}"
        self.decomposed_goals.append({
            "id": goal_id,
            "description": goal,
            "status": status,  # pending, in_progress, completed
            "created_at": datetime.datetime.now().isoformat(),
            "completed_at": None,
            "related_searches": [],
            "related_knowledge": []
        })
        return goal_id
    
    def update_goal_status(self, goal_id: str, status: str) -> None:
        """更新目标状态"""
        for goal in self.decomposed_goals:
            if goal["id"] == goal_id:
                goal["status"] = status
                if status == "completed":
                    goal["completed_at"] = datetime.datetime.now().isoformat()
                break
    
    def add_iteration(self, query: str, results: List[SearchResultItem], goal_id: Optional[str] = None) -> str:
        """添加一次搜索迭代记录"""
        search_id = f"search_{len(self.search_iterations) + 1}"
        search_data = {
            "id": search_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "results": [r.dict() for r in results],  # 存储为字典
            "goal_id": goal_id
        }
        self.search_iterations.append(search_data)
        
        if goal_id:
            for goal in self.decomposed_goals:
                if goal["id"] == goal_id:
                    goal["related_searches"].append(search_id)
                    break
        return search_id
    
    def add_knowledge(self, knowledge_item: KnowledgeItem) -> str:
        """向知识库添加已验证的信息"""
        self.knowledge_base.append(knowledge_item)
        if knowledge_item.goal_id:
            for goal in self.decomposed_goals:
                if goal["id"] == knowledge_item.goal_id:
                    goal["related_knowledge"].append(knowledge_item.key)  # 存储知识的key
                    break
        return knowledge_item.key  # 返回知识的key作为ID
    
    def get_knowledge_item(self, key: str) -> Optional[KnowledgeItem]:
        """根据key获取知识条目"""
        for item in self.knowledge_base:
            if item.key == key:
                return item
        return None

    def update_knowledge_item(self, key: str, **kwargs) -> bool:
        """更新知识条目的属性"""
        for item in self.knowledge_base:
            if item.key == key:
                for k, v in kwargs.items():
                    setattr(item, k, v)
                return True
        return False
    
    def add_hypothesis(self, hypothesis: str, supporting_knowledge_ids: List[str] = None) -> str:
        """添加一个研究假设"""
        hyp_id = f"hyp_{len(self.hypotheses) + 1}"
        self.hypotheses.append({
            "id": hyp_id,
            "hypothesis": hypothesis,
            "status": "pending_verification",  # pending_verification, supported, refuted, needs_more_info
            "supporting_knowledge_ids": supporting_knowledge_ids if supporting_knowledge_ids else [],
            "contradicting_knowledge_ids": []
        })
        return hyp_id
    
    def update_hypothesis_status(self, hyp_id: str, status: str, supporting_knowledge_ids: List[str] = None, contradicting_knowledge_ids: List[str] = None) -> bool:
        """更新假设的状态和相关知识"""
        for hyp in self.hypotheses:
            if hyp["id"] == hyp_id:
                hyp["status"] = status
                if supporting_knowledge_ids:
                    hyp["supporting_knowledge_ids"].extend(supporting_knowledge_ids)
                if contradicting_knowledge_ids:
                    hyp["contradicting_knowledge_ids"].extend(contradicting_knowledge_ids)
                return True
        return False

    def add_contradiction(self, conflicting_knowledge_ids: List[str], description: str) -> str:
        """记录一个检测到的矛盾"""
        contra_id = f"contra_{len(self.contradictions) + 1}"
        self.contradictions.append({
            "id": contra_id,
            "description": description,
            "conflicting_knowledge_ids": conflicting_knowledge_ids,
            "resolution_attempt_description": None,
            "status": "unresolved"  # unresolved, resolving, resolved
        })
        return contra_id
    
    def update_contradiction_status(self, contra_id: str, status: str, resolution_description: Optional[str] = None) -> bool:
        """更新矛盾的状态和解决方案"""
        for contra in self.contradictions:
            if contra["id"] == contra_id:
                contra["status"] = status
                if resolution_description:
                    contra["resolution_attempt_description"] = resolution_description
                return True
        return False

    def get_knowledge_by_recency(self, days: int = 365) -> List[KnowledgeItem]:
        """获取指定时间范围内的知识，按发布日期排序，最新的在前"""
        cutoff_date = datetime.datetime.now() - relativedelta(days=days)
        recent_knowledge = []
        
        for item in self.knowledge_base:
            try:
                if item.publish_date:
                    pub_date = parse_date(item.publish_date)
                    if pub_date >= cutoff_date:
                        recent_knowledge.append(item)
            except:
                pass  # 如果日期解析失败，跳过
        
        return sorted(recent_knowledge, key=lambda x: parse_date(x.publish_date) if x.publish_date else datetime.datetime.min, reverse=True)
    
    def get_state_summary(self) -> str:
        """获取当前状态摘要，用于Agent的上下文"""
        summary = [
            f"TASK: {self.task_description}",
            f"DECOMPOSED GOALS: {len(self.decomposed_goals)} (Completed: {self.get_goal_progress()['completed']})",
            f"SEARCH ITERATIONS: {len(self.search_iterations)}",
            f"KNOWLEDGE ITEMS: {len(self.knowledge_base)}",
            f"HYPOTHESES: {len(self.hypotheses)}",
            f"CONTRADICTIONS: {len(self.contradictions)}",
            f"GENERATION ATTEMPTS: {len(self.history) + (1 if self.current_generation else 0)}",
        ]
        
        # 添加目标状态摘要
        goals_summary = []
        for goal in self.decomposed_goals:
            goals_summary.append(f"- {goal['id']}: {goal['description']} [{goal['status']}]")
        
        if goals_summary:
            summary.append("GOALS:")
            summary.extend(goals_summary)
        
        if self.current_generation:
            summary.append(f"CURRENT GENERATION SCORE: {self.current_score}/10")
            if self.current_criticism:
                summary.append(f"CURRENT CRITICISM: {self.current_criticism[:200]}...")  # 截断批评
        
        return "\n".join(summary)
    
    def get_goal_progress(self) -> Dict[str, int]:
        """获取目标完成进度统计"""
        stats = {"total": len(self.decomposed_goals), "completed": 0, "in_progress": 0, "pending": 0}
        for goal in self.decomposed_goals:
            stats[goal["status"]] = stats.get(goal["status"], 0) + 1
        return stats

    def add_generation(self, content: str, score: float, criticism: str):
        """记录一次生成尝试"""
        self.history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "content": self.current_generation,  # Save previous generation
            "score": self.current_score,
            "criticism": self.current_criticism
        })
        self.current_generation = content
        self.current_score = score
        self.current_criticism = criticism 