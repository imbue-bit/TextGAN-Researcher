import json
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ..models.execution_state import EnhancedExecutionState


class GeneratorAgent:
    """TextGAN-D架构中的生成器(Generator/Prover)角色"""
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """你是一个专业的研究报告生成器。你的任务是基于当前的执行状态和知识库，生成高质量的研究内容。
            
            执行状态摘要:
            {state_summary}
            
            当前任务: {task}
            
            已有的知识库内容:
            {knowledge}
            
            已生成的假设:
            {hypotheses}
            
            如果存在之前的生成被批评，请注意以下批评:
            {criticism}
            
            现在，请生成一个全面、准确、有深度、有条理的研究内容。确保:
            1. 整合所有相关、最新的信息
            2. 明确引用信息来源 (来自知识库的key或原始source)
            3. 避免之前批评中指出的问题
            4. 提供有条理的分析和洞见，并尝试验证或探讨假设
            5. 关注信息的时效性和可信度
            
            你的生成内容:"""
        )
    
    def generate(self, state: EnhancedExecutionState) -> str:
        knowledge_summary = "\n".join([
            f"- {item.key}: {str(item.value)[:150]}... (来源: {item.source}, 日期: {item.publish_date if item.publish_date else '未知'}, 可信度: {item.confidence*100:.1f}%)"
            for item in state.get_knowledge_by_recency(days=365)  # 优先使用最新知识
        ])
        
        hypotheses_summary = "\n".join([
            f"- {h['id']}: {h['hypothesis']} [{h['status']}]" for h in state.hypotheses
        ])

        criticism = state.current_criticism if state.current_criticism else "无之前的批评"
        
        prompt_input = {
            "state_summary": state.get_state_summary(),
            "task": state.task_description,
            "knowledge": knowledge_summary if knowledge_summary else "知识库为空。",
            "hypotheses": hypotheses_summary if hypotheses_summary else "无已生成假设。",
            "criticism": criticism
        }
        
        response = self.llm.invoke([HumanMessage(content=self.prompt.format(**prompt_input))])
        return response.content


class RewarderAgent:
    """TextGAN-D架构中的奖励者(Rewarder)角色"""
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """你是一个专业的内容评分员。你的任务是对生成的研究内容进行快速评分。
            
            评分标准 (1-10分):
            1. 准确性: 内容是否基于事实，没有错误信息
            2. 全面性: 内容是否涵盖了任务的各个方面，并整合了最新知识
            3. 深度: 内容是否提供了深入的分析而非表面的描述
            4. 相关性: 内容是否与任务高度相关
            5. 可信度: 是否合理引用来源，并体现对信息时效性和可靠性的考量
            
            研究任务: {task}
            
            生成的内容:
            {content}
            
            请给出一个1-10的综合评分，并简要解释理由。仅返回JSON格式:
            {{"score": 分数, "reason": "理由"}}"""
        )
    
    def evaluate(self, task: str, content: str) -> Dict[str, Any]:
        response = self.llm.invoke([HumanMessage(content=self.prompt.format(task=task, content=content))])
        try:
            result = json.loads(response.content)
            return {
                "score": float(result["score"]),
                "reason": result["reason"]
            }
        except Exception as e:
            return {
                "score": 5.0,
                "reason": f"评分解析失败: {str(e)}. 原始响应: {response.content}"
            }


class ReviewerAgent:
    """TextGAN-D架构中的审辩者(Reviewer)角色"""
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.prompt = PromptTemplate.from_template(
            """你是一个严格的研究内容审辩者。你的任务是对生成的研究内容进行深度分析和批评。
            
            研究任务: {task}
            执行状态摘要: {state_summary}
            
            生成的内容:
            {content}
            
            初步评分: {score}/10
            初步评价: {reason}
            
            请提供详细的批评，包括:
            1. 内容中的具体问题或不足 (例如：信息不准确，遗漏关键点，逻辑不清晰)
            2. 错误或可能有误的信息，并指出需要进一步验证的部分
            3. 逻辑或结构上的缺陷 (例如：论证不充分，组织混乱)
            4. 遗漏的重要信息或视角，特别是最新或高可信度信息
            5. 对信息时效性和可信度处理的不足
            6. 具体的改进建议，指导生成器进行下一次迭代
            
            你的批评应当具体、可操作，能够指导下一次更好的生成。请以列表形式给出。"""
        )
    
    def critique(self, task: str, state: EnhancedExecutionState, content: str, score: float, reason: str) -> str:
        if score >= 8.0:  # 高分则无需详细批评
            return "内容质量良好，无需详细批评。"
        
        prompt_input = {
            "task": task,
            "state_summary": state.get_state_summary(),
            "content": content,
            "score": score,
            "reason": reason
        }
        response = self.llm.invoke([HumanMessage(content=self.prompt.format(**prompt_input))])
        return response.content 