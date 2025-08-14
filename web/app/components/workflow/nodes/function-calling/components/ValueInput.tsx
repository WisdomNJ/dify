'use client'
import type { FC } from 'react'
import React, { useState } from 'react'
import useAvailableVarList from '../../_base/hooks/use-available-var-list'
import { VarType } from '../../../types'
import type { Var } from '../../../types'
import cn from '@/utils/classnames'
import Input from '@/app/components/workflow/nodes/_base/components/input-support-select-var'

type Props = {
  nodeId: string
  readonly: boolean
  value: string
  onChange: (str: string) => void
  placeholder?: string
}

const ValueInput: FC<Props> = ({
                                 nodeId,
                                 readonly,
                                 value,
                                 onChange,
                                 placeholder = '',
                               }) => {
  const [isFocus, setIsFocus] = useState(false)
  const { availableVars, availableNodesWithParent } = useAvailableVarList(nodeId, {
    onlyLeafNodeVar: false,
    filterVar: (varPayload: Var) => {
      return [VarType.string, VarType.number, VarType.secret].includes(varPayload.type)
    },
  })

  return (
    <div className='flex items-start  space-x-1'>
      <Input
        instanceId='http-api-url'
        className={cn(isFocus ? 'border-components-input-border-active bg-components-input-bg-active shadow-xs' : 'border-components-input-border-hover bg-components-input-bg-normal', 'w-0 grow rounded-lg border px-3 py-[6px]')}
        value={value}
        onChange={onChange}
        readOnly={readonly}
        nodesOutputVars={availableVars}
        availableNodes={availableNodesWithParent}
        onFocusChange={setIsFocus}
        placeholder={placeholder}
        placeholderClassName='!leading-[21px]'
      />
    </div>
  )
}
export default React.memo(ValueInput)
