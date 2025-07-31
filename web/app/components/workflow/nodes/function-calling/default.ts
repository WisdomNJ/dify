import type { NodeDefault } from '../../types'
import { BlockEnum, PromptRole } from '../../types'
import type { FunctionCallNodeType } from './types'
import { ALL_CHAT_AVAILABLE_BLOCKS, ALL_COMPLETION_AVAILABLE_BLOCKS } from '@/app/components/workflow/blocks'
import { ReasoningModeType } from '@/app/components/workflow/nodes/parameter-extractor/types'

const nodeDefault: NodeDefault<FunctionCallNodeType> = {
    defaultValue: {
        query: [],
        model: {
            provider: '',
            name: '',
            mode: 'coder',
            completion_params: {
                temperature: 0.7,
            },
        },
        reasoning_mode: ReasoningModeType.prompt,
        prompt_template: [{
            role: PromptRole.system,
            text: '',
        }],
        context: {
            enabled: false,
            variable_selector: [],
        },
        vision: {
            enabled: false,
        },
    },
    getAvailablePrevNodes(isChatMode: boolean) {
        return isChatMode
            ? ALL_CHAT_AVAILABLE_BLOCKS
            : ALL_COMPLETION_AVAILABLE_BLOCKS.filter(type => type !== BlockEnum.End)
    },
    getAvailableNextNodes(isChatMode: boolean) {
        return isChatMode ? ALL_CHAT_AVAILABLE_BLOCKS : ALL_COMPLETION_AVAILABLE_BLOCKS
    },
    checkValid(payload: FunctionCallNodeType) {
        let isValid = true
        let errorMessages = ''
        if (!payload.query || payload.query.length === 0)
            errorMessages = '输入变量不能为空'
        else
            isValid = false

        return {
            isValid,
            errorMessage: errorMessages,
        }
    },
}
export default nodeDefault
