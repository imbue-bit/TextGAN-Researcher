import json
from langchain.tools import BaseTool

from ..models.execution_state import EnhancedExecutionState


class UpdateGoalStatusTool(BaseTool):
    """更新目标状态工具"""
    name = "update_goal_status"
    description = "Update the status of a specific decomposed goal. Input: JSON string with 'goal_id' and 'status' (e.g., 'completed', 'in_progress', 'pending'). Example input: '{{\"goal_id\": \"goal_1\", \"status\": \"completed\"}}'"
    
    def __init__(self, state: EnhancedExecutionState):
        super().__init__()
        self.state = state
    
    def _run(self, input_json: str) -> str:
        try:
            data = json.loads(input_json)
            goal_id = data["goal_id"]
            status = data["status"]
            self.state.update_goal_status(goal_id, status)
            return f"目标 '{goal_id}' 状态已更新为 '{status}'."
        except Exception as e:
            return f"更新目标状态失败: {str(e)}. 输入: {input_json}" 