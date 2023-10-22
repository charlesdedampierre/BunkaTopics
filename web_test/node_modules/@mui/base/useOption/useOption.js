'use client';

import _extends from "@babel/runtime/helpers/esm/extends";
import * as React from 'react';
import { unstable_useForkRef as useForkRef, unstable_useId as useId } from '@mui/utils';
import { extractEventHandlers } from '../utils/extractEventHandlers';
import { useListItem } from '../useList';
import { useCompoundItem } from '../utils/useCompoundItem';

/**
 *
 * Demos:
 *
 * - [Select](https://mui.com/base-ui/react-select/#hooks)
 *
 * API:
 *
 * - [useOption API](https://mui.com/base-ui/react-select/hooks-api/#use-option)
 */
export function useOption(params) {
  const {
    value,
    label,
    disabled,
    rootRef: optionRefParam,
    id: idParam
  } = params;
  const {
    getRootProps: getListItemProps,
    rootRef: listItemRefHandler,
    highlighted,
    selected
  } = useListItem({
    item: value
  });
  const id = useId(idParam);
  const optionRef = React.useRef(null);
  const selectOption = React.useMemo(() => ({
    disabled,
    label,
    value,
    ref: optionRef,
    id
  }), [disabled, label, value, id]);
  const {
    index
  } = useCompoundItem(value, selectOption);
  const handleRef = useForkRef(optionRefParam, optionRef, listItemRefHandler);
  return {
    getRootProps: (externalProps = {}) => {
      const externalEventHandlers = extractEventHandlers(externalProps);
      return _extends({}, externalProps, getListItemProps(externalEventHandlers), {
        id,
        ref: handleRef,
        role: 'option',
        'aria-selected': selected
      });
    },
    highlighted,
    index,
    selected,
    rootRef: handleRef
  };
}