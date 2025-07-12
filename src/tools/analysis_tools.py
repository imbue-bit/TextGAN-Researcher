import json
import datetime
from typing import List
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

from ..models.base_models import SearchResultItem, KnowledgeItem
from ..models.execution_state import EnhancedExecutionState


class KnowledgeFreshnessAnalysisTool(BaseTool):
    """知识时效性分析工具"""
    name = "analyze_knowledge_freshness"
    description = "Analyze the freshness and relevance of collected knowledge items within a specified time period. Input: time_period (e.g., '3个月', '1年'). Output: A report on knowledge freshness. Example input: '6个月'"
    
    def __init__(self, state: EnhancedExecutionState):
        super().__init__()
        self.state = state
    
    def _run(self, time_period: str = "1年") -> str:
        if "天" in time_period:
            days = int(time_period.replace("天", "").strip())
        elif "月" in time_period:
            days = int(time_period.replace("月", "").strip()) * 30
        elif "年" in time_period:
            days = int(time_period.replace("年", "").strip()) * 365
        else:
            days = 365
        
        recent_knowledge = self.state.get_knowledge_by_recency(days)
        
        if not recent_knowledge:
            return f"在过去的{time_period}内没有找到任何知识条目。"
        
        now = datetime.datetime.now()
        report = [f"过去{time_period}内的知识时效性报告:"]
        report.append(f"找到{len(recent_knowledge)}条相关知识条目。")
        
        age_buckets = {"1个月内": 0, "1-3个月": 0, "3-6个月": 0, "6个月-1年": 0, "1年以上": 0}
        
        for item in recent_knowledge:
            try:
                if item.publish_date:
                    pub_date = parse_date(item.publish_date)
                    age = relativedelta(now, pub_date)
                    
                    if age.years == 0 and age.months == 0 and age.days < 30:
                        age_buckets["1个月内"] += 1
                    elif age.years == 0 and age.months < 3:
                        age_buckets["1-3个月"] += 1
                    elif age.years == 0 and age.months < 6:
                        age_buckets["3-6个月"] += 1
                    elif age.years == 0:
                        age_buckets["6个月-1年"] += 1
                    else:
                        age_buckets["1年以上"] += 1
                else:
                    age_buckets["1年以上"] += 1  # 没有发布日期，视为旧数据
            except:
                age_buckets["1年以上"] += 1
        
        report.append("\n时效性分布:")
        for period, count in age_buckets.items():
            if count > 0:
                report.append(f"- {period}: {count}条 ({count/len(recent_knowledge)*100:.1f}%)")
        
        report.append("\n最新知识条目 (最多5条):")
        for i, item in enumerate(recent_knowledge[:5], 1):
            report.append(f"{i}. {item.key} (来源: {item.source}, 日期: {item.publish_date if item.publish_date else '未知'})")
            report.append(f"   可信度: {item.confidence*100:.1f}%")
            report.append(f"   内容: {str(item.value)[:100]}..." if len(str(item.value)) > 100 else f"   内容: {item.value}")
        
        return "\n".join(report)


class ContradictionDetectionTool(BaseTool):
    """矛盾检测工具"""
    name = "detect_contradictions"
    description = "Analyze the knowledge base for conflicting information. Input: A JSON list of knowledge item keys to check (optional, checks all if empty). Output: A report on detected contradictions. Example input: '[\"AI_benefits\", \"AI_risks\"]'"
    
    def __init__(self, llm: BaseChatModel, state: EnhancedExecutionState):
        super().__init__()
        self.llm = llm
        self.state = state
        self.prompt = PromptTemplate.from_template(
            """你是一个逻辑分析专家。你的任务是审查提供的知识条目，识别其中是否存在相互矛盾的信息。
            
            当前研究任务: {task_description}
            
            知识条目 (key: value, source, publish_date, confidence):
            {knowledge_items_str}
            
            请分析这些知识条目。如果发现矛盾，请描述矛盾点，并列出涉及矛盾的知识条目key。
            如果不存在明显矛盾，请说明。
            
            输出格式 (JSON):
            {{"contradictions": [ {{"description": "矛盾描述", "conflicting_keys": ["key1", "key2"]}} ], "status": "no_contradictions" | "contradictions_found"}}
            """
        )
    
    def _run(self, knowledge_keys_json: str = "[]") -> str:
        try:
            keys_to_check = json.loads(knowledge_keys_json)
            knowledge_to_check = []
            
            if keys_to_check:
                for k in keys_to_check:
                    item = self.state.get_knowledge_item(k)
                    if item:
                        knowledge_to_check.append(item)
            else:  # 如果没有指定key，检查所有知识
                knowledge_to_check = self.state.knowledge_base
            
            if len(knowledge_to_check) < 2:
                return "知识条目不足，无法进行矛盾检测。"
            
            knowledge_items_str = "\n".join([
                f"- {item.key}: {item.value} (来源: {item.source}, 日期: {item.publish_date}, 可信度: {item.confidence})"
                for item in knowledge_to_check
            ])
            
            response = self.llm.invoke([HumanMessage(content=self.prompt.format(
                task_description=self.state.task_description,
                knowledge_items_str=knowledge_items_str
            ))])
            
            analysis_result = json.loads(response.content)
            
            if analysis_result.get("status") == "contradictions_found" and analysis_result.get("contradictions"):
                report = ["检测到以下矛盾:"]
                for contra in analysis_result["contradictions"]:
                    self.state.add_contradiction(contra["conflicting_keys"], contra["description"])
                    report.append(f"- 描述: {contra['description']}")
                    report.append(f"  涉及知识: {', '.join(contra['conflicting_keys'])}")
                return "\n".join(report)
            else:
                return "未检测到明显矛盾。"
        except Exception as e:
            return f"矛盾检测失败: {str(e)}. 原始输入: {knowledge_keys_json}. 原始响应: {response.content}"


class ResolveContradictionTool(BaseTool):
    """解决矛盾工具"""
    name = "resolve_contradiction"
    description = "Analyze a specific contradiction and propose a resolution. This might involve searching for more information, prioritizing sources, or re-evaluating facts. Input: JSON string with 'contradiction_id' and 'method' (e.g., 'search_more', 'prioritize_freshness', 'prioritize_confidence'). Output: A description of the resolution attempt and its outcome. Example input: '{{\"contradiction_id\": \"contra_1\", \"method\": \"prioritize_freshness\"}}'"
    
    def __init__(self, llm: BaseChatModel, state: EnhancedExecutionState):
        super().__init__()
        self.llm = llm
        self.state = state
        self.prompt = PromptTemplate.from_template(
            """你是一个矛盾解决专家。你的任务是分析一个特定的矛盾，并提出一个解决方案。
            
            研究任务: {task_description}
            矛盾描述: {contradiction_description}
            涉及知识条目 (key: value, source, publish_date, confidence):
            {conflicting_knowledge_str}
            
            解决策略: {method}
            
            请根据提供的策略，分析矛盾并提出解决方案。如果需要更多信息，请说明。
            
            输出格式 (JSON):
            {{"resolution_description": "解决方案描述", "status": "resolved" | "needs_more_info" | "unresolved"}}
            """
        )
    
    def _run(self, input_json: str) -> str:
        try:
            data = json.loads(input_json)
            contra_id = data["contradiction_id"]
            method = data["method"]
            
            contradiction = next((c for c in self.state.contradictions if c["id"] == contra_id), None)
            if not contradiction:
                return f"未找到矛盾ID: {contra_id}"
            
            conflicting_knowledge_items = [self.state.get_knowledge_item(k) for k in contradiction["conflicting_keys"] if self.state.get_knowledge_item(k)]
            
            conflicting_knowledge_str = "\n".join([
                f"- {item.key}: {item.value} (来源: {item.source}, 日期: {item.publish_date}, 可信度: {item.confidence})"
                for item in conflicting_knowledge_items
            ])
            
            response = self.llm.invoke([HumanMessage(content=self.prompt.format(
                task_description=self.state.task_description,
                contradiction_description=contradiction["description"],
                conflicting_knowledge_str=conflicting_knowledge_str,
                method=method
            ))])
            
            resolution_result = json.loads(response.content)
            self.state.update_contradiction_status(contra_id, resolution_result["status"], resolution_result["resolution_description"])
            
            # 如果解决了，可以考虑更新相关知识的可信度或标记为已验证
            if resolution_result["status"] == "resolved":
                # 示例：假设解决方案会提高某个知识的可信度或标记为已验证
                for k_key in contradiction["conflicting_keys"]:
                    self.state.update_knowledge_item(k_key, verified=True)
            
            return f"矛盾 '{contra_id}' 尝试解决。结果: {resolution_result['status']}. 描述: {resolution_result['resolution_description']}"
        except Exception as e:
            return f"解决矛盾失败: {str(e)}. 输入: {input_json}. 原始响应: {response.content}"


class HypothesisGenerationTool(BaseTool):
    """假设生成工具"""
    name = "generate_hypothesis"
    description = "Generate a new research hypothesis or question based on the current knowledge base. Input: Optional string describing the area of focus. Output: The generated hypothesis string. Example input: 'impact of climate change on specific crops'"
    
    def __init__(self, llm: BaseChatModel, state: EnhancedExecutionState):
        super().__init__()
        self.llm = llm
        self.state = state
        self.prompt = PromptTemplate.from_template(
            """你是一个富有洞察力的研究员。你的任务是基于当前的知识库和研究任务，提出一个新的、可验证的研究假设或问题。
            
            当前研究任务: {task_description}
            
            已有知识 (最多5条最新知识):
            {knowledge_summary}
            
            当前已分解目标:
            {goals_summary}
            
            请提出一个具体的、有启发性的假设或问题。你的输出应该是假设或问题的文本。
            """
        )
    
    def _run(self, focus_area: str = "") -> str:
        knowledge_summary = "\n".join([
            f"- {item.key}: {str(item.value)[:150]}... (来源: {item.source}, 日期: {item.publish_date})"
            for item in self.state.get_knowledge_by_recency(days=365)[:5]
        ])
        goals_summary = "\n".join([
            f"- {g['id']}: {g['description']} [{g['status']}]" for g in self.state.decomposed_goals
        ])

        response = self.llm.invoke([HumanMessage(content=self.prompt.format(
            task_description=self.state.task_description,
            knowledge_summary=knowledge_summary if knowledge_summary else "无",
            goals_summary=goals_summary if goals_summary else "无",
            focus_area=focus_area
        ))])
        
        hypothesis = response.content
        self.state.add_hypothesis(hypothesis)
        return f"已生成新假设: '{hypothesis}'"


class VerifyHypothesisTool(BaseTool):
    """验证假设工具"""
    name = "verify_hypothesis"
    description = "Attempt to verify or refute a specific hypothesis by searching for supporting or contradicting evidence. Input: 'hypothesis_id' (string). Output: A summary of the verification attempt and updated hypothesis status. Example input: 'hyp_1'"
    
    def __init__(self, llm: BaseChatModel, state: EnhancedExecutionState, web_search_tool):
        super().__init__()
        self.llm = llm
        self.state = state
        self.web_search_tool = web_search_tool  # 依赖搜索工具
        self.prompt = PromptTemplate.from_template(
            """你是一个假设验证专家。你的任务是根据提供的知识和搜索结果，判断一个假设是否被支持或驳斥。
            
            研究任务: {task_description}
            假设: {hypothesis}
            
            当前相关知识:
            {relevant_knowledge_str}
            
            最近搜索结果:
            {search_results_str}
            
            请判断该假设的状态 (supported, refuted, needs_more_info)，并提供详细理由。
            输出格式 (JSON):
            {{"status": "supported" | "refuted" | "needs_more_info", "reason": "详细理由", "supporting_knowledge_keys": [], "contradicting_knowledge_keys": []}}
            """
        )
    
    def _run(self, hypothesis_id: str) -> str:
        try:
            hypothesis = next((h for h in self.state.hypotheses if h["id"] == hypothesis_id), None)
            if not hypothesis:
                return f"未找到假设ID: {hypothesis_id}"
            
            # 1. 自动进行相关搜索 (Agent 内部调用工具)
            search_query = f"evidence for/against '{hypothesis['hypothesis']}'"
            search_results_json = self.web_search_tool._run(search_query)  # 直接调用工具的_run方法
            search_results = [SearchResultItem(**r) for r in json.loads(search_results_json)]
            
            # 2. 从搜索结果中提取新知识并添加到状态
            extracted_knowledge_keys = []
            for sr in search_results:
                # 模拟提取，这里可以更智能地调用 InformationExtractionTool
                # 为了简化，直接将搜索结果内容作为知识
                new_key = f"evidence_{hypothesis_id}_{len(extracted_knowledge_keys)+1}"
                new_knowledge = KnowledgeItem(
                    key=new_key,
                    value=sr.content,
                    source=sr.url,
                    publish_date=sr.publish_date,
                    confidence=sr.confidence,
                    goal_id=hypothesis.get("goal_id")  # 假设可以关联到某个目标
                )
                self.state.add_knowledge(new_knowledge)
                extracted_knowledge_keys.append(new_key)

            # 3. 准备相关知识和搜索结果给LLM进行判断
            relevant_knowledge = [self.state.get_knowledge_item(k) for k in hypothesis["supporting_knowledge_ids"] if self.state.get_knowledge_item(k)]
            relevant_knowledge.extend([self.state.get_knowledge_item(k) for k in extracted_knowledge_keys if self.state.get_knowledge_item(k)])
            
            relevant_knowledge_str = "\n".join([
                f"- {item.key}: {item.value} (来源: {item.source}, 日期: {item.publish_date}, 可信度: {item.confidence})"
                for item in relevant_knowledge
            ])
            search_results_str = "\n".join([f"- {sr.content[:100]}... (来源: {sr.url}, 日期: {sr.publish_date})" for sr in search_results])
            
            response = self.llm.invoke([HumanMessage(content=self.prompt.format(
                task_description=self.state.task_description,
                hypothesis=hypothesis["hypothesis"],
                relevant_knowledge_str=relevant_knowledge_str,
                search_results_str=search_results_str
            ))])
            
            verification_result = json.loads(response.content)
            
            # 4. 更新假设状态
            self.state.update_hypothesis_status(
                hypothesis_id, 
                verification_result["status"],
                supporting_knowledge_ids=verification_result.get("supporting_knowledge_keys", []),
                contradicting_knowledge_ids=verification_result.get("contradicting_knowledge_keys", [])
            )
            
            return f"假设 '{hypothesis_id}' ({hypothesis['hypothesis']}) 已验证。状态: {verification_result['status']}. 理由: {verification_result['reason']}"
        except Exception as e:
            return f"验证假设失败: {str(e)}. 输入: {hypothesis_id}. 原始响应: {response.content}" 