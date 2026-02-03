import re

from ..base import BaseTool

import arxiv

from typing import Optional

class ComplianceChecker(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "compliance_checker"
        self.description = "A tool for checking the compliance of legal documents and business practices with applicable laws and regulations, highlighting areas that may require adjustments."
        self.tool_instruction = f"Instruction: Use {self.name} to {self.description}"

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