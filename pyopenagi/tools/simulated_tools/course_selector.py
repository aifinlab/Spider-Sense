import re

from ..base import BaseTool

import arxiv

from typing import Optional

class CourseSelector(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "course_selector"
        self.description = "A tool for recommending courses and educational institutions based on the student'
        self.tool_instruction = f"Instruction: Use {self.name} to {self.description}"s interests, academic background, and career goals."

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