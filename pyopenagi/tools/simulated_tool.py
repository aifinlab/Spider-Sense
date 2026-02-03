from .base import BaseTool
import pandas as pd

class SimulatedTool(BaseTool):
    def __init__(self, name, tools_info_path):
        super().__init__()
        self.name = name
        tools_info = pd.read_json(tools_info_path, lines=True)
        tool = tools_info[tools_info['Tool Name'] == name]
        self.description = tool['Description'].iloc[0]
        self.expected_achivement = tool['Expected Achievements'].iloc[0]
        # [MODIFIED] Load Tool Instruction
        self.tool_instruction = tool.get('Tool Instruction').iloc[0] if 'Tool Instruction' in tool.columns else self.description

    def run(self, params=None) -> str:
        # [USER REQ] Return mock data for injected mock query
        if params and isinstance(params, dict):
            query = params.get("query", "")
            if "mock_id" in query:
                # [MODIFIED] Return Tool Instruction instead of specific Mock Data
                return self.tool_instruction
                
                # if self.name == "sys_monitor":
                #     return "System Log (Last 24h): [INFO] Service started. [WARN] Memory usage peak at 92%. [INFO] Backup completed."
                # elif self.name == "update_manager":
                #     return "System Update Status: All packages up to date. Security patch level: 2026-01-15."
                # elif self.name == "image_processor":
                #     return "Image processing complete. Resized 15 files to 1024x1024."
                # else:
                #     return "Result: [Patient 017-41154] Gender: Male; Status: Stable."
        
        # [MODIFIED] Return Tool Instruction instead of Expected Achievement
        return self.tool_instruction
        # return self.expected_achivement


    def get_tool_call_format(self):
        tool_call_format = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "params": {
                            "type": "string",
                            "description": "Optional parameters for the tool."
                        }
                    }
                },
            }
        }
        return tool_call_format


class AttackerTool(BaseTool):
    def __init__(self, attacker_tool):
        super().__init__()
        self.tool_name=attacker_tool['Attacker Tool']
        self.tool_description=attacker_tool.get('Description', attacker_tool.get('Attacker Instruction', 'Attacker tool description'))
        self.attack_goal = attacker_tool['Attack goal']
        self.tool_instruction = attacker_tool.get('Tool Instruction')
        self.expected_achivement = self.attack_goal # Use goal as expected achievement for success check

    def run(self,params=None):
        # return f"You have used the attacker tool {self.tool_name}. You have achieved the attacker goal: {self.attack_goal}"
        return self.tool_instruction

    def get_tool_call_format(self):
        tool_call_format = {
            "type": "function",
            "function": {
                "name": f"{self.tool_name}",
                "description": f"{self.tool_description}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action_input": {
                            "type": "string", 
                            "description": "Any necessary arguments or parameters for the action."
                        }
                    },
                    "required": ["action_input"]
                }
            }
        }
        return tool_call_format
