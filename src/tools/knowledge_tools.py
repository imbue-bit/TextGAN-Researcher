import json
import datetime
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ..models.base_models import KnowledgeItem
from ..models.execution_state import EnhancedExecutionState


class InformationExtractionTool(BaseTool):
    """信息提取工具"""
    name: str = "extract_information"
    description: str = "Extract key facts, entities, and data points from raw text (e.g., search results) and structure them into a KnowledgeItem. Input: JSON string with 'text' (the content to extract from), 'source' (original URL or description), 'publish_date' (ISO format, optional), 'confidence' (0.0-1.0, optional), and 'goal_id' (optional). Output: JSON string of the extracted KnowledgeItem. Example input: '{{\"text\": \"AI is rapidly advancing.\", \"source\": \"blog.com\", \"publish_date\": \"2023-01-01\", \"confidence\": 0.8, \"goal_id\": \"goal_1\"}}'"
    
    def __init__(self, llm: BaseChatModel, state: EnhancedExecutionState):
        super().__init__()
        self.llm = llm
        self.state = state
        self.prompt = PromptTemplate.from_template(
            """你是一个专业的信息提取专家。你的任务是从提供的文本中提取关键信息，并将其结构化为 KnowledgeItem。
            
            任务描述：{task_description}
            当前子目标 (如果适用): {current_goal_description}
            
            文本内容：
            {text_content}
            
            请提取最重要的事实、观点或数据，并以 JSON 格式的 KnowledgeItem 返回。
            KnowledgeItem 结构：{{"key": "主题/名称", "value": "提取的具体内容", "source": "来源 URL 或描述", "publish_date": "ISO 格式日期 (如果文本中包含)", "confidence": "0.0-1.0", "goal_id": "关联的子目标 ID"}}
            
            如果无法确定发布日期，请省略 publish_date 字段。如果文本中没有明确的可信度信息，请使用默认值 0.7。
            仅返回 JSON，不要有其他说明。"""
        )
    
    def _run(self, input_json: str) -> str:
        try:
            data = json.loads(input_json)
            text_content = data["text"]
            source = data.get("source", "未知来源")
            publish_date = data.get("publish_date")
            confidence = data.get("confidence", 0.7)
            goal_id = data.get("goal_id")
            
            current_goal_description = "无"
            if goal_id:
                for g in self.state.decomposed_goals:
                    if g["id"] == goal_id:
                        current_goal_description = g["description"]
                        break
            
            response = self.llm.invoke([HumanMessage(content=self.prompt.format(
                task_description=self.state.task_description,
                current_goal_description=current_goal_description,
                text_content=text_content
            ))])
            
            extracted_data = json.loads(response.content)
            
            # 确保 extracted_data 包含所有 KnowledgeItem 必需字段，并设置默认值
            extracted_data["source"] = extracted_data.get("source", source)
            extracted_data["publish_date"] = extracted_data.get("publish_date", publish_date)
            extracted_data["confidence"] = extracted_data.get("confidence", confidence)
            extracted_data["goal_id"] = extracted_data.get("goal_id", goal_id)
            
            knowledge_item = KnowledgeItem(**extracted_data)
            self.state.add_knowledge(knowledge_item)
            return f"成功提取并添加知识：'{knowledge_item.key}' (来源：{knowledge_item.source}, 日期：{knowledge_item.publish_date})"
        except Exception as e:
            return f"信息提取失败：{str(e)}. 原始输入：{input_json}. 原始响应：{response.content}"


class SynthesizeKnowledgeTool(BaseTool):
    """知识合成工具"""
    name: str = "synthesize_knowledge"
    description: str = "Synthesize multiple existing knowledge items into a new, higher-level knowledge item. This is useful for summarizing findings or creating new insights. Input: JSON string with 'knowledge_item_keys' (list of keys to synthesize), 'new_key' (string for the new knowledge item's key), 'goal_id' (optional). Output: The key of the newly synthesized knowledge item. Example input: '{{\"knowledge_item_keys\": [\"AI_benefits\", \"AI_risks\"], \"new_key\": \"AI_overview\", \"goal_id\": \"goal_1\"}}'"
    
    def __init__(self, llm: BaseChatModel, state: EnhancedExecutionState):
        super().__init__()
        self.llm = llm
        self.state = state
        self.prompt = PromptTemplate.from_template(
            """你是一个知识合成专家。你的任务是将提供的多个知识条目进行综合，形成一个新的、更高级别的知识点或总结。
            
            研究任务：{task_description}
            
            要合成的知识条目：
            {knowledge_items_str}
            
            请将这些知识点进行提炼、整合和归纳，形成一个连贯的、有价值的新知识。
            请提供新知识的价值 (value)，并为它提供一个合适的来源 (source) 和可信度 (confidence)。
            
            输出格式 (JSON):
            {{"value": "合成后的新知识内容", "source": "合成来源 (例如：'基于知识合成')", "confidence": "0.0-1.0"}}
            """
        )
    
    def _run(self, input_json: str) -> str:
        try:
            data = json.loads(input_json)
            knowledge_item_keys = data["knowledge_item_keys"]
            new_key = data["new_key"]
            goal_id = data.get("goal_id")
            
            items_to_synthesize = [self.state.get_knowledge_item(k) for k in knowledge_item_keys if self.state.get_knowledge_item(k)]
            
            if not items_to_synthesize:
                return "没有找到可用于合成的知识条目。"
            
            knowledge_items_str = "\n".join([
                f"- {item.key}: {item.value} (来源：{item.source}, 日期：{item.publish_date}, 可信度：{item.confidence})"
                for item in items_to_synthesize
            ])
            
            response = self.llm.invoke([HumanMessage(content=self.prompt.format(
                task_description=self.state.task_description,
                knowledge_items_str=knowledge_items_str
            ))])
            
            synthesis_result = json.loads(response.content)
            
            new_knowledge_item = KnowledgeItem(
                key=new_key,
                value=synthesis_result["value"],
                source=synthesis_result.get("source", "知识合成"),
                publish_date=datetime.datetime.now().isoformat(),  # 合成知识的日期为当前日期
                confidence=synthesis_result.get("confidence", 0.8),  # 默认较高可信度
                goal_id=goal_id,
                verified=True  # 合成知识视为已验证
            )
            self.state.add_knowledge(new_knowledge_item)
            return f"成功合成新知识：'{new_key}'"
        except Exception as e:
            return f"知识合成失败：{str(e)}. 输入：{input_json}. 原始响应：{response.content}"


class ViewKnowledgeBaseTool(BaseTool):
    """查看知识库工具"""
    name: str = "view_knowledge_base"
    description: str = "View a summary of the current knowledge base. Input: Optional JSON string with 'limit' (int) for number of items, 'recency_days' (int) for recent items. Output: A summary string of knowledge items. Example input: '{{\"limit\": 5, \"recency_days\": 30}}'"
    
    def __init__(self, state: EnhancedExecutionState):
        super().__init__()
        self.state = state
    
    def _run(self, input_json: str = "{}") -> str:
        try:
            params = json.loads(input_json)
            limit = params.get("limit", 10)
            recency_days = params.get("recency_days")

            knowledge_items = []
            if recency_days:
                knowledge_items = self.state.get_knowledge_by_recency(recency_days)
            else:
                knowledge_items = self.state.knowledge_base
            
            if not knowledge_items:
                return "知识库当前为空。"
            
            summary = ["当前知识库内容 (最多显示{}条):".format(limit)]
            for i, item in enumerate(knowledge_items[:limit]):
                summary.append(f"{i+1}. {item.key} (来源：{item.source}, 日期：{item.publish_date if item.publish_date else '未知'}, 可信度：{item.confidence*100:.1f}%, 验证：{item.verified})")
                summary.append(f"   内容：{str(item.value)[:150]}...")
            return "\n".join(summary)
        except Exception as e:
            return f"查看知识库失败：{str(e)}. 输入：{input_json}" 