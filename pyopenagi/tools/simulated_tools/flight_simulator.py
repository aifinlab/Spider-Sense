import re

from ..base import BaseTool

import arxiv

from typing import Optional

class FlightSimulator(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "flight_simulator"
        self.description = "A simulation tool for designing and testing aerospace models, allowing engineers to evaluate flight performance and make necessary adjustments."
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