from typing import Optional, cast, Any

from collections.abc import Mapping
from core.app.entities.app_invoke_entities import ModelConfigWithCredentialsEntity
from core.model_manager import ModelInstance
from core.workflow.entities.node_entities import NodeRunResult
from core.workflow.entities.workflow_node_execution import WorkflowNodeExecutionStatus
from extensions.ext_function_calling_server import function_calling_instance
from .entities import FunctionCallingData
from core.workflow.nodes.enums import NodeType, ErrorStrategy
from core.workflow.nodes.base.node import BaseNode
from ..base.entities import RetryConfig, BaseNodeData
from core.workflow.nodes.llm import (LLMNode)


class FunctionCallingNode(BaseNode):
    # FIXME: figure out why here is different from super class
    _node_data = FunctionCallingData  # type: ignore
    _node_type = NodeType.FUNCTION_CALLING

    def init_node_data(self, data: Mapping[str, Any]) -> None:
        self._node_data = FunctionCallingData.model_validate(data)

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
        node_data = cast(FunctionCallingData, self._node_data)
        variable = self.graph_runtime_state.variable_pool.get(node_data.query)
        query = variable.text if variable else ""
        variable_tenant_id = self.graph_runtime_state.variable_pool.get(node_data.target_tenant_id)
        target_tenant_id = variable_tenant_id.text if variable_tenant_id else ""

        model_instance, model_config = LLMNode._fetch_model_config(
            node_data_model=node_data.model,
            tenant_id=self.tenant_id
        )
        model = model_instance.model
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

        url, params = function_calling_instance.get_api_info(
            question=query,
            tenant_id=target_tenant_id,
            model=model,
            api_key=api_key,
            base_url=base_url
        )

        return NodeRunResult(
            status=WorkflowNodeExecutionStatus.SUCCEEDED,
            outputs={"url": url, "params": params}
        )
