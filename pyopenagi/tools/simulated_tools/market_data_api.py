import re

from ..base import BaseTool

import arxiv

from typing import Optional

class MarketDataApi(BaseTool):
    def __init__(self):
        super().__init__()
        self.name = "market_data_api"
        self.description = "A tool that provides real-time access to market data, including stock prices, trends, and economic indicators, to support financial analysis and decision-making."
        self.tool_instruction = f"Instruction: Use {self.name} to {self.description}"

    def run(self, params) -> str:
        # return f"You have successfully provided real-time access to market data, including stock prices, trends, and economic indicators, to support financial analysis and decision-making."
        return self.tool_instruction

    def get_tool_call_format(self):
        tool_call_format = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": None,
            }
        }
        return tool_call_format