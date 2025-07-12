from langchain.tools import BaseTool


class GenerateContentTool(BaseTool):
    """GAN 循环工具 (现在接受'purpose'参数)"""
    name = "generate_content"
    description = "Generate research content based on current execution state and knowledge base. This tool triggers the Generator-Rewarder-Reviewer cycle. Input: A brief description of the purpose of this generation (e.g., 'initial draft', 'final report', 'summary for goal X'). Output: A summary of the generation process and feedback."
    
    def __init__(self, agent_instance):
        super().__init__()
        self.agent_instance = agent_instance  # 传入DeepResearchAgent实例
        
    def _run(self, purpose: str = "研究报告") -> str:
        self.agent_instance.callback_manager.on_tool_start({"name": self.name, "input": purpose})
        
        # Generator 生成内容
        generated_content = self.agent_instance.generator.generate(self.agent_instance.state)
        self.agent_instance.callback_manager.on_tool_end(f"Generator produced content for '{purpose}'.")
        
        # Rewarder 评估内容
        evaluation = self.agent_instance.rewarder.evaluate(
            self.agent_instance.state.task_description, 
            generated_content
        )
        self.agent_instance.callback_manager.on_tool_end(f"Rewarder scored content: {evaluation['score']}/10. Reason: {evaluation['reason']}")
        
        # Reviewer 批评内容
        criticism = self.agent_instance.reviewer.critique(
            self.agent_instance.state.task_description,
            self.agent_instance.state,
            generated_content,
            evaluation["score"],
            evaluation["reason"]
        )
        self.agent_instance.callback_manager.on_tool_end(f"Reviewer provided criticism: {criticism[:150]}...")
        
        # 更新执行状态
        self.agent_instance.state.add_generation(generated_content, evaluation["score"], criticism)
        
        return f"已生成并评估内容 (目的: {purpose}, 分数: {evaluation['score']}/10)。批评: {criticism[:200]}...\n\n你可以使用 'view_current_generation' 查看完整内容，或根据批评继续迭代。"


class ViewCurrentGenerationTool(BaseTool):
    """查看当前生成内容工具"""
    name = "view_current_generation"
    description = "View the most recently generated research content. Input: None. Output: The full content string."
    
    def __init__(self, agent_instance):
        super().__init__()
        self.agent_instance = agent_instance
        
    def _run(self, _: str = "") -> str:
        if not self.agent_instance.state.current_generation:
            return "No content has been generated yet."
        return self.agent_instance.state.current_generation 