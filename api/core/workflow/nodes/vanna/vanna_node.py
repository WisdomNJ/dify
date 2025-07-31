from typing import Any, Optional, cast, Mapping

from configs import dify_config
from core.app.entities.app_invoke_entities import ModelConfigWithCredentialsEntity
from core.model_manager import ModelInstance

from core.workflow.entities.node_entities import NodeRunResult
from core.workflow.nodes.enums import NodeType, ErrorStrategy
from extensions.utils.vanna_text2sql import VannaServer
from core.workflow.entities.workflow_node_execution import WorkflowNodeExecutionStatus
from extensions.utils.vanna_text2sql_tool import handle_sql
from .entities import VannaNodeData
from core.workflow.nodes.base.node import BaseNode
from ..base.entities import RetryConfig, BaseNodeData
from core.workflow.nodes.llm import (LLMNode)


class Config:
    def __init__(self, supplier):
        self.embedding_supplier = "SiliconFlow"
        self.milvus_uri = dify_config.VANNA_MILVUS_URI
        self.milvus_database = dify_config.VANNA_MILVUS_DATABASE
        self.embedding_host = dify_config.VANNA_EMBEDDING_HOST
        self.embedding_model = dify_config.VANNA_EMBEDDING_MODEL
        self.embedding_type = dify_config.VANNA_EMBEDDING_TYPE
        self.supplier = supplier
        self.sql_type = 'postgres'
        self.sql_config = {
            "host": dify_config.VANNA_DB_HOST,
            "dbname": dify_config.VANNA_DB_DATABASE,
            "user": dify_config.VANNA_DB_USERNAME,
            "password": dify_config.VANNA_DB_PASSWORD,
            "port": dify_config.VANNA_DB_PORT
        }


vn_instances = {}


def get_vanna_server(key, vanna_config):
    if key not in vn_instances:
        config = Config(key)
        # 合并配置
        combined_config = {**config.__dict__, **config.sql_config, **vanna_config}
        vn_instances[key] = VannaServer(combined_config)
    return vn_instances[key]


class VannaNode(BaseNode):
    # FIXME: figure out why here is different from super class
    _node_data = VannaNodeData  # type: ignore
    _node_type = NodeType.VANNA

    def init_node_data(self, data: Mapping[str, Any]) -> None:
        self._node_data = VannaNodeData.model_validate(data)

    def _get_error_strategy(self) -> Optional[ErrorStrategy]:
        return self._node_data.error_strategy

    def _get_retry_config(self) -> RetryConfig:
        return self._node_data.retry_config

    def _get_title(self) -> str:
        return self._node_data.title

    def _get_description(self) -> Optional[str]:
        return self._node_data.desc

    def _get_default_value_dict(self) -> dict[str, Any]:
        return self._node_data.default_value_dict

    def get_base_node_data(self) -> BaseNodeData:
        return self._node_data

    _model_instance: Optional[ModelInstance] = None
    _model_config: Optional[ModelConfigWithCredentialsEntity] = None

    @classmethod
    def get_default_config(cls, filters: Optional[dict] = None) -> dict:
        return {
            "model": {
                "prompt_templates": {
                    "completion_model": {
                        "conversation_histories_role": {"user_prefix": "Human", "assistant_prefix": "Assistant"},
                        "stop": ["Human:"],
                    }
                }
            }
        }

    @classmethod
    def version(cls) -> str:
        return "1"

    def _run(self):
        node_data = cast(VannaNodeData, self._node_data)
        variable = self.graph_runtime_state.variable_pool.get(node_data.query)
        variable_tenant_id = self.graph_runtime_state.variable_pool.get(node_data.target_tenant_id)
        query = variable.text if variable else ""
        target_tenant_id = variable_tenant_id.text if variable_tenant_id else ""

        model_instance, model_config = LLMNode._fetch_model_config(
            node_data_model=node_data.model,
            tenant_id=self.tenant_id
        )
        # 'tongyi' 通义 'openai' openai 'ollama' ollama 'deepseek' deepseek
        llm_type = model_instance.provider.rsplit('/')[-1]

        api_key = ''
        base_url = ''
        match model_config.provider:
            case 'langgenius/deepseek/deepseek':
                api_key = model_instance.credentials.get('api_key')
                base_url = model_instance.credentials.get('endpoint_url')
            case 'langgenius/tongyi/tongyi':
                api_key = model_instance.credentials.get('dashscope_api_key')
                base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            case 'langgenius/ollama/ollama':
                base_url = model_instance.credentials.get('base_url')

        cache_kay = model_config.provider + api_key if api_key else base_url
        vanna_config = {
            "llm_type": llm_type,
            "model": model_instance.model,
            "api_key": api_key,
            "ollama_host": base_url
        }

        cache_data = get_vanna_server(cache_kay, vanna_config)
        # 提问获取sql和结果
        sql = cache_data.generate_sql(question=query, tenant_id=target_tenant_id)
        # 对生成的SQL做处理
        sql = handle_sql(sql=sql)
        return NodeRunResult(
            status=WorkflowNodeExecutionStatus.SUCCEEDED,
            outputs={"output": sql}
        )
