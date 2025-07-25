import type { VannaNodeType } from './types'
import useNodeCrud from '@/app/components/workflow/nodes/_base/hooks/use-node-crud'
import {
    useNodesReadOnly,
} from '@/app/components/workflow/hooks'
import { useCallback, useRef } from 'react'
import produce from 'immer'
import type { ValueSelector, Var } from '@/app/components/workflow/types'
import { VarType } from '@/app/components/workflow/types'
import useOneStepRun from '@/app/components/workflow/nodes/_base/hooks/use-one-step-run'
import useConfigVision from '@/app/components/workflow/hooks/use-config-vision'
import useAvailableVarList from '@/app/components/workflow/nodes/_base/hooks/use-available-var-list'

const useConfig = (id: string, payload: VannaNodeType) => {
    const { nodesReadOnly: readOnly } = useNodesReadOnly()
    const { inputs, setInputs } = useNodeCrud<VannaNodeType>(id, payload)
    const inputRef = useRef(inputs)

    // model
    const model = inputs.model || {
        provider: '',
        name: '',
        mode: 'chat',
        completion_params: {
            temperature: 0.7,
        },
    }

    // single run
    const {
        isShowSingleRun,
        hideSingleRun,
        getInputVars,
        runningStatus,
        handleRun,
        handleStop,
        runInputData,
        runInputDataRef,
        setRunInputData,
        runResult,
    } = useOneStepRun<VannaNodeType>({
        isPaused: false,
        isRunAfterSingleRun: false,
        id,
        data: inputs,
        defaultRunInputData: {
            'query': '',
            '#files#': [],
        },
    })

    const {
        isVisionModel,
    } = useConfigVision(model, {
        payload: inputs.vision,
        onChange: (newPayload) => {
            const newInputs = produce(inputs, (draft) => {
                draft.vision = newPayload
            })
            setInputs(newInputs)
        },
    })

    const filterVisionInputVar = useCallback((varPayload: Var) => {
        return [VarType.file, VarType.arrayFile].includes(varPayload.type)
    }, [])

    const {
        availableVars: availableVisionVars,
    } = useAvailableVarList(id, {
        onlyLeafNodeVar: false,
        filterVar: filterVisionInputVar,
    })

    const handleModelChanged = useCallback((model: { provider: string; modelId: string; mode?: string }) => {
        const newInputs = produce(inputRef.current, (draft) => {
            draft.model.provider = model.provider
            draft.model.name = model.modelId
            draft.model.mode = model.mode!
        })
        setInputs(newInputs)
    }, [setInputs])

    const handleCompletionParamsChange = useCallback((newParams: Record<string, any>) => {
        const newInputs = produce(inputs, (draft) => {
            draft.model.completion_params = newParams
        })
        setInputs(newInputs)
    }, [inputs, setInputs])

    const handleInputVarChange = useCallback((newInputVar: ValueSelector | string) => {
        const newInputs = produce(inputs, (draft) => {
            draft.query = newInputVar as ValueSelector || []
        })
        setInputs(newInputs)
    }, [inputs, setInputs])

    const filterVar = useCallback((varPayload: Var) => {
        return [VarType.string].includes(varPayload.type)
    }, [])

    const varInputs = getInputVars([inputs.instruction])
    const inputVarValues = (() => {
        const vars: Record<string, any> = {}
        Object.keys(runInputData)
            .forEach((key) => {
                vars[key] = runInputData[key]
            })
        return vars
    })()

    const setInputVarValues = useCallback((newPayload: Record<string, any>) => {
        setRunInputData(newPayload)
    }, [setRunInputData])

    const visionFiles = runInputData['#files#']
    const setVisionFiles = useCallback((newFiles: any[]) => {
        setRunInputData({
            ...runInputDataRef.current,
            '#files#': newFiles,
        })
    }, [runInputDataRef, setRunInputData])

    return {
        readOnly,
        inputs,
        filterVar,
        isShowSingleRun,
        hideSingleRun,
        getInputVars,
        runningStatus,
        handleRun,
        handleStop,
        runInputData,
        runInputDataRef,
        setRunInputData,
        isVisionModel,
        runResult,
        varInputs,
        inputVarValues,
        setInputVarValues,
        handleModelChanged,
        handleInputVarChange,
        handleCompletionParamsChange,
        availableVisionVars,
        visionFiles,
        setVisionFiles,
    }
}

export default useConfig
