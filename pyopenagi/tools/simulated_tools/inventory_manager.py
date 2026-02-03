import re

from ..base import BaseTool

import arxiv

from typing import Optional

class InventoryManager(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "inventory_manager"
        self.description = "A tool for managing product inventory, tracking stock levels, and optimizing product listings to ensure availability and efficient supply chain management."
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