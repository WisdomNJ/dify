import type {
    CommonNodeType,
    Memory,
    ModelConfig,
    PromptItem,
    ValueSelector,
    VisionSetting,
} from '@/app/components/workflow/types'
import type { Param, ReasoningModeType } from '@/app/components/workflow/nodes/parameter-extractor/types'

export type FunctionCallNodeType = CommonNodeType & {
    model: ModelConfig
    query: ValueSelector
    prompt_template: PromptItem[] | PromptItem,
    instruction: string
    reasoning_mode: ReasoningModeType
    parameters: Param[]
    memory?: Memory
    context: {
        enabled: boolean
        variable_selector: ValueSelector
    }
    vision: {
        enabled: boolean
        configs?: VisionSetting
    }
}
