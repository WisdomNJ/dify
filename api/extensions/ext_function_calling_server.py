import json

from openai.types import Embedding

from configs import dify_config
from typing import List, Dict

from dify_app import DifyApp
from pymilvus import MilvusClient
from pymilvus.model.base import BaseEmbeddingFunction
from flask import jsonify, request
from openai import OpenAI
import numpy as np
from extensions.utils.search_tool import api_desc_match
import extensions.utils.date_utils as date_utils

# 自定义嵌入式模型（适配milvus向量数据库）
class CustomEmbeddingFunction(BaseEmbeddingFunction):

    def __init__(self):
        self.embed_model = dify_config.VANNA_EMBEDDING_MODEL
        self.embedding_model = OpenAI(
            base_url=dify_config.VANNA_EMBEDDING_HOST,
            api_key=dify_config.VANNA_EMBEDDING_API_KEY,
        )

    def __call__(self, texts: List[str]):
        self._encode(texts)

    def _encode(self, texts: list[str]) -> list[Embedding]:
        embeddings = self.embedding_model.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [embedding.embedding for embedding in embeddings.data]

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        response = self._encode(queries)
        return [np.array(embedding) for embedding in response]

custom_embedding_instance = CustomEmbeddingFunction()


class FunctionCallingServer:
    def __init__(self):
        self.embedding_function = custom_embedding_instance
        self.milvus_client = MilvusClient(
            uri=dify_config.VANNA_MILVUS_URI,
            db_name=dify_config.VANNA_MILVUS_DATABASE,
            user=dify_config.VANNA_MILVUS_USER,
            password=dify_config.VANNA_MILVUS_PASSWORD,
        )

    def get_related_func(self, question: str) -> list:

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 8},
        }

        embeddings = self.embedding_function.encode_queries([question])

        res = self.milvus_client.search(
            collection_name="vannafunc",
            anns_field="vector",
            data=embeddings,
            limit=10,
            output_fields=["url", "description", "params", "ext", "id", "type", "content", "ext_prompt", "word_keys",
                           "synonym", "word"],
            search_params=search_params
        )
        res = res[0]
        list_func = []
        for doc in res:
            print(doc["distance"])
            params = json.loads(doc["entity"]["params"]) if doc["entity"]["params"] else {}
            url = doc["entity"]["url"]
            description = doc["entity"]["description"]
            ext = doc["entity"]["ext"]
            type = doc["entity"]["type"]
            content = doc["entity"]["content"]
            word_keys = doc["entity"]["word_keys"]
            synonym = doc["entity"]["synonym"]
            word = doc["entity"]["word"]
            ext_prompt = doc["entity"]["ext_prompt"]
            id = doc["entity"]["id"]
            list_func.append({
                "id": id,
                "params": params,
                "url": url,
                "type": type,
                "content": content,
                "word_keys": word_keys,
                "synonym": synonym,
                "word": word,
                "ext_prompt": ext_prompt,
                "ext": ext,
                "description": description
            })
        return list_func

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

    def get_system_message(self, ext_prompts: str):

        prompt = f"""
        你是一个Ai助手，你需要借助工具，回答用户问题，任务如下：
        1. 根据用户问句，精准匹配接口文档中唯一一个最相关接口。
        2. 提取或推理接口所需所有参数值，参数类型必须与接口文档完全一致。
        3. 参数提取时，支持基本类型转换（数字、字符串），不支持复杂类型推断。
        4. 在匹配时，使用严格字符串匹配，不支持模糊匹配。
        5. 今天是{date_utils.get_today()}，本周日期{date_utils.get_this_week_start()}至{date_utils.get_this_week_end()}，上半年日期：01-01至06-30，前年是{date_utils.this_last_year(2)}年
        6. 所有时间段条件：如果按月查询，开始时间：月初，结束时间：月底，如果是年，开始时间：年初，完成时间：年底，如果是季度，开始时间：季度初，完成时间季度底
        {ext_prompts}
        """
        return {
            "role": "system",
            "content": prompt
        }

    def prepare_tools(self, tools: List[Dict]):
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["id"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": tool["params"]
                    }
                }
            }
            for tool in tools
        ]

    def get_run_api(self, api_info: dict, question: str, tenant_id: str) -> dict:
        type = api_info["type"]
        # 获取API 信息
        ext = api_info["ext"]
        url = api_info["url"]
        description = api_info["description"]
        params = api_info["params"]
        content = api_info["content"]
        if not ext:
            ext = {}
        if type == "sql":
            return {
                "url": url,
                "api_status": 1,
                "description": description,
                "body": {
                    "sql": content,
                    "params": params,
                    "tenantId": tenant_id,
                    "ext": ext
                }
            }
        else:
            return {
                "url": url,
                "api_status": 1,
                "description": description,
                "body": {
                    **params,
                    "content": content,
                    "tenantId": tenant_id,
                    "ext": ext,
                    "question": question
                }
            }

    def filter_api_info(self, question, funcs):
        if len(funcs) > 0:
            for func in funcs:
                word_keys = func["word_keys"]
                description = func["description"]
                word = func["word"]
                synonym = func["synonym"]
                if word_keys:
                    word_keys = json.loads(word_keys)
                    target_required = word_keys["required"]
                    target_un_required = word_keys["un_required"]
                    ok = api_desc_match(question_text=question,
                                        name=description,
                                        target_required=target_required,
                                        target_un_required=target_un_required,
                                        word=word,
                                        synonym=synonym)
                    if ok:
                        return [func]
        return []

    def get_api_info_by_model(self, question:str, tenant_id:str, model:str, api_key:str, base_url:str, funcs:list) -> ( str, dict ):

        wanted_keys = {"description", "params", "id", "ext_prompt"}
        api_prompt_list = [{k: v for k, v in f.items() if k in wanted_keys} for f in funcs]
        ext_prompt_list = [f"工具:`{v["id"]}` : \n{v["ext_prompt"]}" for v in api_prompt_list]
        ext_prompts = "\n".join(ext_prompt_list)
        system_message = self.get_system_message(ext_prompts=ext_prompts)

        tools = self.prepare_tools(api_prompt_list)

        messages = [system_message, self.user_message(question)]

        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.2,
            max_tokens=8192,
        )

        if response.choices[0].finish_reason != "tool_calls":
            return None, None, {}
        tool_calls = response.choices[0].message.tool_calls

        api_info = None
        if len(tool_calls) > 0:
            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                api_info = {"id": name, "params": arguments}
                print(name, arguments)

        if api_info:
            result = next((f for f in funcs if f.get("id") == api_info["id"]), None)
            api_info = {**result, **api_info}

        # 验证环节
        if "id" not in api_info or not api_info["id"]:
            return None, None,{}

        # 根据API信息，执行接口
        run_api_info = self.get_run_api(api_info=api_info, question=question, tenant_id=tenant_id)

        return run_api_info["url"], run_api_info["description"], run_api_info["body"]

    def get_api_info(self, question:str, tenant_id:str, model:str, api_key:str, base_url:str) -> ( str, str,dict ):

        # 获取所有的问句
        funcs = self.get_related_func(question=question)
        # 分词过滤
        funcs = self.filter_api_info(question=question, funcs=funcs)
        if len(funcs) == 0:
            return None,None, {}

        return self.get_api_info_by_model(
            question=question,
            tenant_id=tenant_id,
            model=model,
            api_key=api_key,
            base_url=base_url,
            funcs=funcs
        )

    def get_api_info_test(self, question:str, tenant_id:str, model:str, api_key:str, base_url:str,) -> ( str,str, dict ):
        # 获取所有的问句
        funcs = self.get_related_func(question=question)
        # 分词过滤
        filter_funcs = self.filter_api_info(question=question, funcs=funcs)

        if len(filter_funcs) == 0:
            filter_funcs = funcs

        if len(funcs) == 0:
            return None, {}

        return self.get_api_info_by_model(
            question=question,
            tenant_id=tenant_id,
            model=model,
            api_key=api_key,
            base_url=base_url,
            funcs=filter_funcs
        )

function_calling_instance = FunctionCallingServer()


def init_app(app: DifyApp):
    @app.route('/api/fast_generate_sql2', methods=['GET'])
    def get_api_info():
        question = request.args.get('question')
        tenant_id = request.args.get('tenant_id')
        url, desc, params = function_calling_instance.get_api_info(
            question=question,
            model=dify_config.VANNA_MODEL,
            api_key=dify_config.VANNA_API_KEY,
            base_url=dify_config.VANNA_OLLAMA_HOST,
            tenant_id=tenant_id
        )

        return jsonify(
            {
                "url": url,
                "desc" : desc,
                "params": params,
            }
        ), 200

    @app.route('/api/text2api_test', methods=['GET'])
    def get_api_info_test():
        question = request.args.get('question')
        tenant_id = request.args.get('tenant_id')
        url, desc, params = function_calling_instance.get_api_info_test(
            question=question,
            model=dify_config.VANNA_MODEL,
            api_key=dify_config.VANNA_API_KEY,
            base_url=dify_config.VANNA_OLLAMA_HOST,
            tenant_id=tenant_id
        )

        return jsonify(
            {
                "url": url,
                "desc" : desc,
                "params": params,
            }
        ), 200


