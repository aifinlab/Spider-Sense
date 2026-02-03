import re

from ..base import BaseTool

import arxiv

from typing import Optional

class PrescriptionManager(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "prescription_manager"
        self.description = "A tool for managing and issuing prescriptions, ensuring that medications are correctly prescribed based on the patient'
        self.tool_instruction = f"Instruction: Use {self.name} to {self.description}"s medical history and current condition."

    def run(self, params) -> str:
        # return f"{self.description}."
        return self.tool_instruction

    def get_tool_call_format(self):
        tool_call_format = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": None
            }
        }
        return tool_call_format