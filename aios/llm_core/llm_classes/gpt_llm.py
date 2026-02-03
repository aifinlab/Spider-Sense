import re
from .base_llm import BaseLLM
import time

# could be dynamically imported similar to other models
from openai import OpenAI

import openai

from pyopenagi.utils.chat_template import Response
import json

class GPTLLM(BaseLLM):

    def __init__(self, llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 1024,
                 log_mode: str = "console"):
        super().__init__(llm_name,
                         max_gpu_memory,
                         eval_device,
                         max_new_tokens,
                         log_mode)

    def load_llm_and_tokenizer(self) -> None:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        if "qwen" in self.model_name.lower():
            self.model = OpenAI(
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") # 兼容旧逻辑
            )
        else:
            # Default to OpenRouter for GPT models if OpenRouter key is present
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                 self.model = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=openrouter_key
                )
            else:
                # Fallback to standard OpenAI or other env config
                self.model = OpenAI()
                
        self.tokenizer = None

    def parse_tool_calls(self, tool_calls):
        if tool_calls:
            parsed_tool_calls = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                arguments_str = tool_call.function.arguments
                if not arguments_str or arguments_str.strip() == "":
                    function_args = {}
                else:
                    try:
                        function_args = json.loads(arguments_str)
                    except json.JSONDecodeError:
                        function_args = {"raw_arguments": arguments_str}
                parsed_tool_calls.append(

                    {
                        "name": function_name,
                        "parameters": function_args
                    }
                )
            return parsed_tool_calls
        return None

    def process(self,
            agent_process,
            temperature=0.0
        ):
        # ensures the model is the current one
        # assert re.search(r'gpt|qwen', self.model_name, re.IGNORECASE)

        """ wrapper around openai api """
        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        messages = agent_process.query.messages
        # print(messages)
        self.logger.log(
            f"{agent_process.agent_name} is switched to executing.\n",
            level = "executing"
        )
        time.sleep(2)
        
        # [OpenRouter Mapping]
        target_model = self.model_name
        if "claude-3-5-sonnet" in self.model_name:
            target_model = "anthropic/claude-3.5-sonnet"
        
        try:
            response = self.model.chat.completions.create(
                model=target_model,
                messages = messages,
                tools = agent_process.query.tools,
                # tool_choice = "required" if agent_process.query.tools else None,
                max_tokens = self.max_new_tokens,
                seed = 0,
                temperature = temperature,
            )
            response_message = response.choices[0].message.content
            tool_calls = self.parse_tool_calls(
                response.choices[0].message.tool_calls
            )
            # print(tool_calls)
            # print(response.choices[0].message)
            # Extract token usage
            token_usage = None
            if hasattr(response, 'usage') and response.usage:
                token_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }

            agent_process.set_response(
                Response(
                    response_message = response_message,
                    tool_calls = tool_calls,
                    token_usage = token_usage
                )
            )
        except openai.APIConnectionError as e:
            agent_process.set_response(
                Response(
                    response_message = f"Server connection error: {e.__cause__}"
                )
            )
        except openai.RateLimitError as e:
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI RATE LIMIT error {e.status_code}: (e.response)"
                )
            )
        except openai.BadRequestError as e:
            print(f"DEBUG: OpenAI BAD REQUEST Details: {e.body}")
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI BAD REQUEST error {e.status_code}: (e.response)"
                )
            )
        except openai.APIStatusError as e:
            print(f"DEBUG: OpenAI STATUS ERROR Details: {e.body}")
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI STATUS error {e.status_code}: (e.response)"
                )
            )
        except Exception as e:
            agent_process.set_response(
                Response(
                    response_message = f"An unexpected error occurred: {e}"
                )
            )

        agent_process.set_status("done")
        agent_process.set_end_time(time.time())
