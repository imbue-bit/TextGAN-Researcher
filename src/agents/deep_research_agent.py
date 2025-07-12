import datetime
from typing import List, Optional
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain import hub

from ..models.execution_state import EnhancedExecutionState
from ..tools import (
    EnhancedWebSearchTool, GoalDecompositionTool, InformationExtractionTool,
    KnowledgeFreshnessAnalysisTool, ContradictionDetectionTool, ResolveContradictionTool,
    HypothesisGenerationTool, VerifyHypothesisTool, SynthesizeKnowledgeTool,
    UpdateGoalStatusTool, ViewKnowledgeBaseTool, GenerateContentTool, ViewCurrentGenerationTool
)
from .textgan_agents import GeneratorAgent, RewarderAgent, ReviewerAgent


class DeepResearchAgent:
    """Deep Research Agent 主类"""
    def __init__(self, llm: BaseChatModel, verbose: bool = True, 
                 callbacks: Optional[List[BaseCallbackHandler]] = None):
        self.llm = llm
        self.state: Optional[EnhancedExecutionState] = None
        self.verbose = verbose
        
        self.callback_manager = CallbackManager(callbacks if callbacks else [])
        
        # 初始化TextGAN-D角色
        self.generator = GeneratorAgent(llm)
        self.rewarder = RewarderAgent(llm)
        self.reviewer = ReviewerAgent(llm)
        
        self.tools = []  # 工具将在initialize中动态生成
        self.agent_executor = None

    def initialize(self, task_description: str, task_id: Optional[str] = None) -> None:
        """
        初始化一个新的研究任务。每次调用都会创建一个全新的状态。
        task_id 可选，用于为当前任务提供一个标识符。
        """
        # 每次都创建一个全新的状态
        if task_id is None:
            task_id = f"task_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.state = EnhancedExecutionState(task_id=task_id, task_description=task_description)
        print(f"--- 开始新任务: {self.state.task_description} (ID: {self.state.task_id}) ---")
        
        # 动态创建工具实例，传入state和llm
        # 注意：VerifyHypothesisTool 需要 EnhancedWebSearchTool 实例
        web_search_tool_instance = EnhancedWebSearchTool()
        self.tools = [
            web_search_tool_instance,  # 确保搜索工具被实例化并添加到列表
            GoalDecompositionTool(self.llm, self.state),
            InformationExtractionTool(self.llm, self.state),
            KnowledgeFreshnessAnalysisTool(self.state),
            ContradictionDetectionTool(self.llm, self.state),
            ResolveContradictionTool(self.llm, self.state),  # 新增工具
            HypothesisGenerationTool(self.llm, self.state),
            VerifyHypothesisTool(self.llm, self.state, web_search_tool_instance),  # 新增工具，依赖搜索工具
            SynthesizeKnowledgeTool(self.llm, self.state),  # 新增工具
            UpdateGoalStatusTool(self.state),
            ViewKnowledgeBaseTool(self.state),
            GenerateContentTool(self),  # 传入自身实例以便调用GAN角色和更新状态
            ViewCurrentGenerationTool(self)
        ]
        
        # --- 从 LangChain Hub 拉取 ReAct 提示模板 ---
        # 使用 react-json 模板，它支持工具的JSON输入/输出
        # 注意: hub.pull 可能会进行网络请求，确保网络连接
        react_json_prompt = hub.pull("hwchase17/react-json")

        # 自定义系统消息，注入我们的研究哲学和当前状态
        custom_system_message = SystemMessage(
            content=f"""你是一个高度智能且经验丰富的 DeepResearch 智能体。你的任务是进行深入、批判性的研究，并生成高质量的报告。
            
            你的核心目标是成为一个"知识发现者"，而不仅仅是"信息检索器"。这意味着你需要：
            1. **主动分解目标**: 将复杂任务拆解为可管理的子目标。
            2. **迭代式搜索与学习**: 根据初步结果动态调整搜索策略。
            3. **批判性信息处理**: 评估信息来源、时效性和可信度，识别并解决矛盾。
            4. **知识整合与提炼**: 将分散的信息整合为结构化知识，并生成新假设。
            5. **自我纠正与优化**: 通过生成-评估-批评的循环不断完善产出。
            
            当前执行状态摘要:
            {self.state.get_state_summary()}
            
            请严格遵循以下流程和注意事项：
            - **思考 (Thought)**: 在采取任何行动之前，请仔细思考。
                - **审视当前状态**: 检查 `decomposed_goals` (特别是 `pending` 或 `in_progress` 的目标)、`knowledge_base` (是否有知识空白或过时信息)、`hypotheses` (是否有待验证的假设)、`contradictions` (是否有未解决的矛盾)。
                - **确定优先级**: 哪个子目标最重要？哪个知识点最需要补充或验证？是否有矛盾需要立即解决？是否是时候生成内容了？
                - **规划下一步行动**: 根据你的分析，选择最合适的工具和输入。
            - **启动任务**: 如果尚未分解任务，首先使用 `decompose_goal` 工具将主任务分解为子目标。
            - **子目标管理**: 优先处理 `pending` 或 `in_progress` 的子目标。完成一个子目标后，使用 `update_goal_status` 将其标记为 `completed`。
            - **信息收集**: 使用 `web_search` 收集原始数据。
            - **知识提取**: 从搜索结果中利用 `extract_information` 提取关键事实并添加到知识库。务必包含 `source`, `publish_date`, `confidence`。
            - **知识管理**:
                - 定期使用 `analyze_knowledge_freshness` 检查知识的时效性。
                - 使用 `detect_contradictions` 发现知识库中的矛盾，并使用 `resolve_contradiction` 尝试解决。
                - 使用 `generate_hypothesis` 提出新的研究假设，并使用 `verify_hypothesis` 进行验证。
                - 使用 `synthesize_knowledge` 将多个相关知识点合成更高级别的洞察。
            - **内容生成**: 当你认为某个子目标或整个任务的知识足够时，使用 `generate_content` 生成研究内容。根据批评进行迭代。
            - **最终报告**: 当所有子目标都完成，并且你对生成的内容满意时，使用 `final_answer` 提交最终报告。
            
            请记住，你的每一次 'Thought' 都应该清晰地反映你当前的策略和下一步的计划。
            """
        )

        # 将自定义系统消息插入到 hub 模板的 messages 列表的最前面
        # react-json 模板的 messages 通常只有 HumanMessage 和 AIMessage
        # 我们需要将 SystemMessage 放在最前面
        messages = [custom_system_message] + react_json_prompt.messages

        self.agent = create_react_agent(
            self.llm,
            self.tools,
            PromptTemplate.from_messages(messages)  # 使用 PromptTemplate.from_messages
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            callback_manager=self.callback_manager  # 将回调管理器传递给执行器
        )
    
    def run(self, task_description: str, task_id: Optional[str] = None) -> str:
        """
        运行Deep Research Agent完成研究任务。
        task_id 可选，用于为当前任务提供一个标识符，每次运行都会从新状态开始。
        """
        self.initialize(task_description, task_id)
        
        self.callback_manager.on_chain_start({"name": "DeepResearchAgent", "input": {"task": task_description}})
        
        # 初始调用，将用户任务传递给Agent
        # 注意: ReAct 模板通常期望 'input' 和 'agent_scratchpad'
        result = self.agent_executor.invoke({
            "input": task_description,
            "agent_scratchpad": ""  # 初始为空
        })
        
        self.callback_manager.on_chain_end({"output": result["output"]})
        
        return result["output"]
    
    def get_research_process_summary(self) -> str:
        """获取研究过程的摘要统计"""
        if not self.state:
            return "尚未开始任何研究任务"
        
        goal_stats = self.state.get_goal_progress()
        
        summary = [
            "=== 研究过程摘要 ===",
            f"- 任务: {self.state.task_description}",
            f"- 任务ID: {self.state.task_id}",
            f"- 子目标: 共{goal_stats['total']}个 (已完成: {goal_stats['completed']}, 进行中: {goal_stats['in_progress']}, 待处理: {goal_stats['pending']})",
            f"- 搜索迭代: {len(self.state.search_iterations)}次",
            f"- 收集知识条目: {len(self.state.knowledge_base)}条",
            f"- 已生成假设: {len(self.state.hypotheses)}个",
            f"- 检测到矛盾: {len(self.state.contradictions)}个"
        ]
        
        recent_1y = len(self.state.get_knowledge_by_recency(365))
        recent_6m = len(self.state.get_knowledge_by_recency(180))
        recent_3m = len(self.state.get_knowledge_by_recency(90))
        
        summary.append("\n知识时效性统计:")
        total_knowledge = len(self.state.knowledge_base)
        if total_knowledge > 0:
            summary.append(f"- 过去3个月内: {recent_3m}条 ({recent_3m/total_knowledge*100:.1f}%)")
            summary.append(f"- 过去6个月内: {recent_6m}条 ({recent_6m/total_knowledge*100:.1f}%)")
            summary.append(f"- 过去1年内: {recent_1y}条 ({recent_1y/total_knowledge*100:.1f}%)")
        else:
            summary.append("- 知识库为空，无时效性数据。")
        
        if self.state.current_generation:
            summary.append(f"\n最终生成内容评分: {self.state.current_score}/10")
            if self.state.current_criticism:
                summary.append(f"最终批评: {self.state.current_criticism[:200]}...")

        return "\n".join(summary) 
