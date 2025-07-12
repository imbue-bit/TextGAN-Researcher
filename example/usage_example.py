"""
Deep Research Agent 使用示例
演示如何使用 DeepResearchAgent 进行智能研究
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import StdOutCallbackHandler

from src.agents.deep_research_agent import DeepResearchAgent


def main():
    """主函数：演示 DeepResearchAgent 的使用"""
    
    # 确保设置了 OpenAI API 密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 初始化LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)  # 使用更经济的模型进行测试
    
    # 创建一个回调处理器，将Agent的思考过程输出到控制台
    handler = StdOutCallbackHandler()
    
    # --- 第一次运行：开始新任务 ---
    # 每次都创建一个新的 DeepResearchAgent 实例
    deep_research_agent_1 = DeepResearchAgent(llm, verbose=True, callbacks=[handler])
    
    print("--- 第一次运行：开始新研究任务 ---")
    task_description_1 = "分析2024年全球AI芯片市场的主要趋势、竞争格局以及未来发展预测。"
    research_result_1 = deep_research_agent_1.run(task_description_1, task_id="AI_Chip_Market_2024")
    
    print("\n--- 第一次运行完成 ---")
    print(deep_research_agent_1.get_research_process_summary())
    print("\n=== 最终研究报告 (第一次运行) ===\n")
    print(research_result_1)
    
    # --- 第二次运行：开始另一个全新的任务 ---
    # 再次创建一个新的 DeepResearchAgent 实例
    deep_research_agent_2 = DeepResearchAgent(llm, verbose=True, callbacks=[handler])
    
    print("\n\n--- 第二次运行：开始另一个全新的研究任务 ---")
    task_description_2 = "探讨生成式AI对创意产业（如艺术、音乐、写作）的潜在影响和挑战。"
    research_result_2 = deep_research_agent_2.run(task_description_2, task_id="Generative_AI_Creative_Impact")
    
    print("\n--- 第二次运行完成 ---")
    print(deep_research_agent_2.get_research_process_summary())
    print("\n=== 最终研究报告 (第二次运行) ===\n")
    print(research_result_2)
    
    # 验证两个 Agent 实例的状态是独立的
    print(f"\nAgent 1 的任务ID: {deep_research_agent_1.state.task_id}")
    print(f"Agent 1 的知识库条目数: {len(deep_research_agent_1.state.knowledge_base)}")
    
    print(f"\nAgent 2 的任务ID: {deep_research_agent_2.state.task_id}")
    print(f"Agent 2 的知识库条目数: {len(deep_research_agent_2.state.knowledge_base)}")


def simple_example():
    """简单示例：基本使用"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("请设置 OPENAI_API_KEY 环境变量")
        return
    
    # 初始化
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)
    agent = DeepResearchAgent(llm, verbose=True)
    
    # 运行研究任务
    task = "分析电动汽车市场的发展趋势"
    result = agent.run(task)
    
    print("=== 研究结果 ===")
    print(result)
    
    print("\n=== 研究过程摘要 ===")
    print(agent.get_research_process_summary())


if __name__ == "__main__":
    # 运行完整示例
    main()
    
    # 或者运行简单示例
    # simple_example() 