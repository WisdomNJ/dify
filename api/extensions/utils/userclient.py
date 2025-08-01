import asyncio
import json

from typing import List, Dict
from openai import OpenAI


class UserClient:

    def __init__(self, model='qwen2.5-coder-32b-instruct'):
        self.openai_client = OpenAI(
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
            api_key='sk-7b0c509055994271b30bbe25f6347066',
        )

#         prompt = f"""
# 你是一个Ai助手，你需要借助工具，回答用户问题，任务如下：
# 1. 根据用户问句，精准匹配接口文档中唯一一个最相关接口。
# 2. 提取或推理接口所需所有参数值，参数类型必须与接口文档完全一致。若参数缺失或类型不符，返回null。
# 3. 若问句中关键词未完全匹配任何接口params，返回null。
# 4. 若匹配多个接口，返回最符合的一个。
# 5. 今天是2025-07-31，本周日期2025-07-28至2025-08-03，上半年日期：01-01至06-30
#    前年：2023年
#    所有时间段条件：如果按月查询，开始时间：月初，结束时间：月底，如果是年，开始时间：年初，完成时间：年底，如果是季度，开始时间：季度初，完成时间季度底，如果是本周，开始时间：2025-07-28，结束时间：2025-08-03
# 8. 参数提取时，支持基本类型转换（数字、字符串），不支持复杂类型推断。
# 9. 在匹配时，使用严格字符串匹配，不支持模糊匹配。
# 工具:`34f79e67-5129-48b0-96b3-5ff4e3ed5593-func`:
#     - 只能按照年份查询，其他日期或者时间（比如月、周、日、季度），不能使用该工具
#     - 如果用户问题中提到具体人名、客户、项目名称，即使提到了年份，也不能使用该工具。
# 工具:`50fa4474-0bc9-410b-9dc6-11536e47f5cb-func`:
#     - 如果按月查询，开始时间：月初，结束时间：月底，如果是年，开始时间：年初，完成时间：年底，如果是季度，开始时间：季度初，完成时间季度底，如果是本周，开始时间：2025-07-28，结束时间：2025-08-03，2025年上半年日期：2025-01-01至2025-06-30
#     - 如果用户问题中提到具体人名、客户、项目名称，即使提到了年份，也不能使用该工具。
#         """
        self.tools = []
        self.model = model

    def get_system_message(self, ext_prompts:str):
        prompt = f"""
你是一个Ai助手，你需要借助工具，回答用户问题，任务如下：
1. 根据用户问句，精准匹配接口文档中唯一一个最相关接口。
2. 提取或推理接口所需所有参数值，参数类型必须与接口文档完全一致。
3. 参数提取时，支持基本类型转换（数字、字符串），不支持复杂类型推断。
4. 在匹配时，使用严格字符串匹配，不支持模糊匹配。
5. 今天是2025-07-31，本周日期2025-07-28至2025-08-03，上半年日期：01-01至06-30，前年是2023年
6. 所有时间段条件：如果按月查询，开始时间：月初，结束时间：月底，如果是年，开始时间：年初，完成时间：年底，如果是季度，开始时间：季度初，完成时间季度底，如果是本周，开始时间：2025-07-28，结束时间：2025-08-03
{ext_prompts}
        """
        return {
            "role": "system",
            "content": prompt
        }

    def prepare_tools(self,tools:List[Dict]):

        return [
            {
                "type": "function",
                "function": {
                    "name": tool["id"],
                    "description": tool["description"],
                    "parameters": {
                        "properties" : tool["params"]
                    }
                }
            }
            for tool in tools
        ]

    def chat(self, question:str, api_list :  List[Dict]):

        ext_prompt_list = [ f"工具:`{v["id"]}` : \n{v["ext_prompt"]}"  for v in api_list]
        ext_prompts = "\n".join(ext_prompt_list)
        system_message = self.get_system_message(ext_prompts=ext_prompts)

        tools = self.prepare_tools(api_list)
        user_message = {
            "role": "user",
            "content": question,
        }

        messages = []
        messages.append(system_message)
        messages.append(user_message)

        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )
        print("*************",response.choices[0].finish_reason)
        if response.choices[0].finish_reason != "tool_calls":
            return response.choices[0].message
        tool_calls = response.choices[0].message.tool_calls

        api_info = None
        if len(tool_calls) > 0:
            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                api_info = {"id" : name,"params" : arguments}
                print(name,arguments)

        return api_info
