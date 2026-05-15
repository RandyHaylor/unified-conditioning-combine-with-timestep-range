// Dynamic-slot frontend extension for ConditioningMergeWithTimestepRanges.
//
// Maintains exactly one trailing empty CONDITIONING input slot. When the user
// connects the last empty slot, a new empty slot appears (plus its three
// widgets: _start, _end, _weight). When the user disconnects a trailing slot,
// trailing empties beyond one are removed (plus their widgets).
//
// Only depends on ComfyUI core (`../../scripts/app.js`) and LiteGraph methods
// `addInput` / `removeInput` / `addWidget` and the `inputs[]` / `widgets[]`
// arrays. No third-party imports.
//
// Pattern reference (not imported from): rgthree-comfy's any_switch.js.

import { app } from "../../scripts/app.js";

const NODE_TYPE_NAME_FOR_THIS_EXTENSION = "ConditioningMergeWithTimestepRanges";

const STABILIZE_DEBOUNCE_DELAY_MILLISECONDS = 32;

const CONDITIONING_INPUT_NAME_REGEX = /^conditioning_(\d+)$/;

function findHighestConditioningSlotIndexFromExistingInputs(node) {
  let highest_slot_index_found = 0;
  for (const input_descriptor of node.inputs || []) {
    const match = (input_descriptor && input_descriptor.name)
      ? input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX)
      : null;
    if (match) {
      const slot_index = parseInt(match[1], 10);
      if (slot_index > highest_slot_index_found) {
        highest_slot_index_found = slot_index;
      }
    }
  }
  return highest_slot_index_found;
}

function findLastConnectedConditioningSlotInfo(node) {
  let last_connected_slot_input_index_in_inputs_array = -1;
  let last_connected_slot_number_extracted_from_name = 0;
  for (let input_index_in_inputs_array = 0; input_index_in_inputs_array < (node.inputs || []).length; input_index_in_inputs_array++) {
    const input_descriptor = node.inputs[input_index_in_inputs_array];
    const match = input_descriptor && input_descriptor.name
      ? input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX)
      : null;
    if (!match) continue;
    const is_connected = input_descriptor.link != null && input_descriptor.link !== undefined;
    if (is_connected) {
      last_connected_slot_input_index_in_inputs_array = input_index_in_inputs_array;
      last_connected_slot_number_extracted_from_name = parseInt(match[1], 10);
    }
  }
  return {
    last_connected_input_array_index: last_connected_slot_input_index_in_inputs_array,
    last_connected_slot_number: last_connected_slot_number_extracted_from_name,
  };
}

function countConditioningInputSlotsOnNode(node) {
  let count = 0;
  for (const input_descriptor of node.inputs || []) {
    if (input_descriptor && input_descriptor.name && CONDITIONING_INPUT_NAME_REGEX.test(input_descriptor.name)) {
      count++;
    }
  }
  return count;
}

function addOneConditioningInputSlotAndItsWidgetTriple(node, slot_number_to_use) {
  node.addInput(`conditioning_${slot_number_to_use}`, "CONDITIONING");
  node.addWidget(
    "number",
    `conditioning_${slot_number_to_use}_start`,
    0.0,
    () => {},
    { min: 0.0, max: 1.0, step: 0.001, precision: 3 },
  );
  node.addWidget(
    "number",
    `conditioning_${slot_number_to_use}_end`,
    1.0,
    () => {},
    { min: 0.0, max: 1.0, step: 0.001, precision: 3 },
  );
  node.addWidget(
    "number",
    `conditioning_${slot_number_to_use}_weight`,
    1.0,
    () => {},
    { min: 0.0, max: 10.0, step: 0.01, precision: 2 },
  );
}

function removeLastConditioningInputSlotAndItsWidgetTriple(node) {
  if (!node.inputs || node.inputs.length === 0) return;
  const last_input_index_in_inputs_array = node.inputs.length - 1;
  const last_input_descriptor = node.inputs[last_input_index_in_inputs_array];
  const match = last_input_descriptor && last_input_descriptor.name
    ? last_input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX)
    : null;
  if (!match) return;
  const slot_number = match[1];

  node.removeInput(last_input_index_in_inputs_array);

  // Remove the three widgets named conditioning_<slot_number>_(start|end|weight).
  // Identify by name; remove from end to start to keep indices valid.
  const widget_name_suffixes_to_remove = ["_weight", "_end", "_start"];
  for (const suffix of widget_name_suffixes_to_remove) {
    const widget_name_to_find = `conditioning_${slot_number}${suffix}`;
    const widget_index_in_widgets_array = (node.widgets || []).findIndex(
      (widget_descriptor) => widget_descriptor && widget_descriptor.name === widget_name_to_find,
    );
    if (widget_index_in_widgets_array >= 0) {
      node.widgets.splice(widget_index_in_widgets_array, 1);
    }
  }
}

function stabilizeDynamicSlotsOnConditioningMergeNode(node) {
  // We want exactly one trailing empty slot after the last connected one.
  // If there is no connected slot, we want exactly one empty slot total
  // (the always-present conditioning_1).
  const total_conditioning_slots_currently = countConditioningInputSlotsOnNode(node);
  const last_info = findLastConnectedConditioningSlotInfo(node);
  const desired_total_conditioning_slots = Math.max(1, last_info.last_connected_slot_number + 1);

  // Remove trailing empty slots beyond desired count (only remove from end,
  // never the middle — gaps are fine and let the user keep an unused slot).
  while (countConditioningInputSlotsOnNode(node) > desired_total_conditioning_slots) {
    const last_input_descriptor = node.inputs[node.inputs.length - 1];
    const is_conditioning_slot = last_input_descriptor
      && last_input_descriptor.name
      && CONDITIONING_INPUT_NAME_REGEX.test(last_input_descriptor.name);
    const is_unconnected = !last_input_descriptor.link;
    if (is_conditioning_slot && is_unconnected) {
      removeLastConditioningInputSlotAndItsWidgetTriple(node);
    } else {
      break;
    }
  }

  // Add new trailing empty slots until we reach desired count.
  while (countConditioningInputSlotsOnNode(node) < desired_total_conditioning_slots) {
    const next_slot_number = findHighestConditioningSlotIndexFromExistingInputs(node) + 1;
    addOneConditioningInputSlotAndItsWidgetTriple(node, next_slot_number);
  }

  node.setDirtyCanvas(true, true);
}

function scheduleStabilizationOnNode(node) {
  if (node.__conditioning_merge_stabilize_timer_id != null) {
    clearTimeout(node.__conditioning_merge_stabilize_timer_id);
  }
  node.__conditioning_merge_stabilize_timer_id = setTimeout(() => {
    node.__conditioning_merge_stabilize_timer_id = null;
    stabilizeDynamicSlotsOnConditioningMergeNode(node);
  }, STABILIZE_DEBOUNCE_DELAY_MILLISECONDS);
}

app.registerExtension({
  name: "ConditioningMergeWithTimestepRanges.DynamicSlots",
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== NODE_TYPE_NAME_FOR_THIS_EXTENSION) return;

    const original_on_node_created_function = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const original_result = original_on_node_created_function
        ? original_on_node_created_function.apply(this, arguments)
        : undefined;
      // On fresh node creation, conditioning_1 already exists via INPUT_TYPES.required.
      // Schedule a stabilize so we always have one trailing empty slot.
      scheduleStabilizationOnNode(this);
      return original_result;
    };

    const original_on_connections_change_function = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (slot_io_type, slot_index, is_connected, link_info, io_slot_descriptor) {
      const original_result = original_on_connections_change_function
        ? original_on_connections_change_function.apply(this, arguments)
        : undefined;
      // Only react to input-side changes (LiteGraph.INPUT === 1).
      if (slot_io_type === 1 /* LiteGraph.INPUT */) {
        scheduleStabilizationOnNode(this);
      }
      return original_result;
    };

    const original_on_configure_function = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (serialized_node_data) {
      const original_result = original_on_configure_function
        ? original_on_configure_function.apply(this, arguments)
        : undefined;
      // After loading a saved workflow, ensure stabilization runs so trailing
      // empty slot exists.
      scheduleStabilizationOnNode(this);
      return original_result;
    };
  },
});
