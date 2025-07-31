import json
from configs import dify_config
from typing import List

from dify_app import DifyApp
from pymilvus import MilvusClient
from pymilvus.model.base import BaseEmbeddingFunction
from flask import jsonify, request
from openai import OpenAI
import ollama
import numpy as np


# 自定义嵌入式模型（适配milvus向量数据库）
class CustomEmbeddingFunction(BaseEmbeddingFunction):

    def __init__(self):
        self.embed_model = dify_config.VANNA_EMBEDDING_MODEL
        self.embedding_model = ollama.Client(dify_config.VANNA_EMBEDDING_HOST)
        self.keep_alive = None
        self.ollama_options = {}
        self.num_ctx = self.ollama_options.get('num_ctx', 2048)

    def __call__(self, texts: List[str]):
        self._encode(texts)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        return [self.embedding_model.embeddings(
            model=self.embed_model,
            prompt=text,
            options=self.ollama_options,
            keep_alive=self.keep_alive
        )["embedding"] for text in texts]

    def encode_queries(self, queries: List[str]) -> List[np.array]:
        embeddings = self._encode(queries)
        return [np.array(embedding) for embedding in embeddings]


class FunctionCallingServer:
    def __init__(self):
        self.embedding_function = CustomEmbeddingFunction()
        self.milvus_client = MilvusClient(
            uri=dify_config.VANNA_MILVUS_URI,
            db_name=dify_config.VANNA_MILVUS_DATABASE,
            user=dify_config.VANNA_MILVUS_USER,
            password=dify_config.VANNA_MILVUS_PASSWORD,
        )

    def get_related_func(self, question: str) -> list:

        embeddings = self.embedding_function.encode_queries([question])

        res = self.milvus_client.search(
            collection_name="vannafunc",
            anns_field="vector",
            data=embeddings,
            limit=20,
            output_fields=["url", "description", "params", "ext", "id", "type", "content"],
            search_params={"metric_type": "COSINE", "params": {"nprobe": 8}}
        )
        list_func = []
        for doc in res[0]:
            print(doc["distance"])
            params = json.loads(doc["entity"]["params"])
            url = doc["entity"]["url"]
            description = doc["entity"]["description"]
            ext = doc["entity"]["ext"]
            type = doc["entity"]["type"]
            content = doc["entity"]["content"]
            id = doc["entity"]["id"]
            list_func.append({
                "id": id,
                "params": params,
                "url": url,
                "type": type,
                "content": content,
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

    def get_api_info(self, question, model, api_key, base_url) -> dict:
        # 获取所有的问句
        funcs = self.get_related_func(question=question)
        if len(funcs) == 0:
            return {}

        wanted_keys = {"description", "params", "id"}
        api_prompt_list = [{k: v for k, v in f.items() if k in wanted_keys} for f in funcs]
        # 将字典转换为 JSON 字符串
        json_str = json.dumps(api_prompt_list, ensure_ascii=False)
        prompt = f"""
            你是一个接口匹配助手，任务是：

            1. 根据接口描述，选出与用户问句最相关的一个接口
            2. 提取或推理出接口所需的参数值
            3. 给出最终的函数调用格式
            4. 今天是2025-07-27
            5. 匹配精度要高，参数必须完全匹配，匹配不到返回错误信息
            6. 注意参数的类型，要与params内保持一致
            接口文档如下：

            {json_str}
            请输出json格式如下：
            {{
                "id": "主键",
                "description": "接口说明",
                "params": {{ param1: 2025 , param2: "字符串"}}
            }}
            - params是参数，参数为空显示空字符串
        """
        message_prompt = [self.system_message(prompt)]
        message_prompt.append(self.user_message(question))
        # 初始化 Ollama 的 OpenAI 接口客户端
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=model,
            messages=message_prompt,
            temperature=0.2,
            max_tokens=8192,
        )

        filtered_text = response.choices[0].message.content.strip()
        cleaned_json_str = filtered_text.replace('```json', '').replace('```', '').strip()
        cleaned_json_str = cleaned_json_str.strip().strip('`')
        parsed_dict: dict = json.loads(cleaned_json_str)
        result = next((f for f in funcs if f.get("id") == parsed_dict["id"]), None)
        api_info = {**result, **parsed_dict}
        print(api_info)
        return api_info


function_calling_instance = FunctionCallingServer()


def init_app(app: DifyApp):

    @app.route('/api/fast_generate_sql2', methods=['GET'])
    def get_api_info():
        question = request.args.get('question')
        result = function_calling_instance.get_api_info(
            question=question,
            model=dify_config.VANNA_MODEL,
            api_key=dify_config.VANNA_API_KEY,
            base_url=dify_config.VANNA_OLLAMA_HOST
        )

        return jsonify(
            {
                "result": result,
            }
        ), 200
