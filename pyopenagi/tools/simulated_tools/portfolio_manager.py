import re

from ..base import BaseTool

import arxiv

from typing import Optional

class PortfolioManager(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "portfolio_manager"
        self.description = "A tool for managing investment portfolios, allowing the analysis of asset allocation, risk assessment, and performance tracking to optimize investment strategies."
        self.tool_instruction = f"Instruction: Use {self.name} to {self.description}"

    def run(self, params) -> str:
        # return f"You have successfully managed investment portfolios, allowing the analysis of asset allocation, risk assessment, and performance tracking to optimize investment strategies."
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