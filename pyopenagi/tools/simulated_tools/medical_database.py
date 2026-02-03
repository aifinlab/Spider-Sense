import re

from ..base import BaseTool

import arxiv

from typing import Optional

class MedicalDatabase(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "medical_database"
        self.description = "A comprehensive database of medical information, including diseases, symptoms, treatments, and medications, to assist in diagnosis and treatment planning."
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