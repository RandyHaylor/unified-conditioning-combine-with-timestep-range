// Dynamic-slot frontend extension for ConditioningMergeWithTimestepRanges.
//
// Behavior:
//   - Each conditioning input slot has three widgets (start, end, weight) and
//     a small canvas-drawn header label "── slot N ──" above its triple to
//     visually group them with the corresponding input.
//   - The slot list auto-expands and auto-collapses on connection changes:
//       * Connecting the trailing empty slot adds a new trailing empty slot.
//       * Disconnecting any slot (whether trailing or middle) removes that
//         slot plus its widget triple plus its header. Exactly one trailing
//         empty slot is always maintained.
//   - Widget values persist across save/load:
//       * `serialize_widgets = true` is set on the node.
//       * `configure(info)` is overridden to rebuild widget triples (and
//         their non-serializing headers) matching the restored slot count
//         BEFORE LiteGraph populates widget values from info.widgets_values
//         positionally (reference: rgthree power_lora_loader.js:21,51-70).
//
// Depends only on ComfyUI core (`../../scripts/app.js`) and LiteGraph core
// methods (addInput / removeInput / addWidget / setDirtyCanvas, plus the
// inputs[] / widgets[] arrays). No third-party imports.

import { app } from "../../scripts/app.js";

const NODE_TYPE_NAME_FOR_THIS_EXTENSION = "ConditioningMergeWithTimestepRanges";

const STABILIZE_DEBOUNCE_DELAY_MILLISECONDS = 32;

const CONDITIONING_INPUT_NAME_REGEX = /^conditioning_(\d+)$/;

const LITEGRAPH_INPUT_SLOT_IO_TYPE_ENUM_VALUE = 1;

// -------------- slot accounting --------------

function findHighestConditioningSlotIndexFromExistingInputs(node) {
  let highest_slot_index_found = 0;
  for (const input_descriptor of node.inputs || []) {
    const match = (input_descriptor && input_descriptor.name)
      ? input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX)
      : null;
    if (match) {
      const slot_index = parseInt(match[1], 10);
      if (slot_index > highest_slot_index_found) highest_slot_index_found = slot_index;
    }
  }
  return highest_slot_index_found;
}

function indexOfTrailingEmptyConditioningSlotInInputsArrayOrMinusOne(node) {
  if (!node.inputs || node.inputs.length === 0) return -1;
  const last_index_in_inputs_array = node.inputs.length - 1;
  const last_input_descriptor = node.inputs[last_index_in_inputs_array];
  if (!last_input_descriptor || !last_input_descriptor.name) return -1;
  if (!CONDITIONING_INPUT_NAME_REGEX.test(last_input_descriptor.name)) return -1;
  if (last_input_descriptor.link != null) return -1;
  return last_index_in_inputs_array;
}

// -------------- widget add / remove --------------

function addOneSlotHeaderLabelWidget(node, slot_number_to_use) {
  // Non-serializing label widget that visually groups the slot's triple
  // with its input slot. `options.serialize = false` keeps it out of
  // widgets_values so positional widget restoration on load stays aligned.
  const slot_header_label_widget = {
    name: `__header_for_slot_${slot_number_to_use}`,
    type: "custom",
    value: "",
    options: { serialize: false },
    draw(canvas_context, owning_node, widget_width_pixels, y_top_pixels, widget_height_pixels) {
      canvas_context.save();
      canvas_context.fillStyle = "#9aa";
      canvas_context.font = "bold 11px Arial, sans-serif";
      canvas_context.textBaseline = "middle";
      const label_text = `── slot ${slot_number_to_use} ──`;
      const label_text_x_pixels = 12;
      const label_text_y_pixels = y_top_pixels + widget_height_pixels / 2;
      canvas_context.fillText(label_text, label_text_x_pixels, label_text_y_pixels);
      canvas_context.restore();
    },
    computeSize(available_widget_width_pixels) {
      return [available_widget_width_pixels, 16];
    },
  };
  if (!node.widgets) node.widgets = [];
  node.widgets.push(slot_header_label_widget);
  return slot_header_label_widget;
}

function addOneWidgetTripleForSlot(node, slot_number_to_use) {
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

function addOneConditioningInputSlotAndItsWidgetTripleAndHeader(node, slot_number_to_use) {
  node.addInput(`conditioning_${slot_number_to_use}`, "CONDITIONING");
  addOneSlotHeaderLabelWidget(node, slot_number_to_use);
  addOneWidgetTripleForSlot(node, slot_number_to_use);
}

function removeWidgetsByNameOnNode(node, list_of_widget_names_to_remove) {
  if (!node.widgets) return;
  for (const widget_name_to_find of list_of_widget_names_to_remove) {
    const widget_index_in_widgets_array = node.widgets.findIndex(
      (widget_descriptor) => widget_descriptor && widget_descriptor.name === widget_name_to_find,
    );
    if (widget_index_in_widgets_array >= 0) {
      node.widgets.splice(widget_index_in_widgets_array, 1);
    }
  }
}

function removeConditioningSlotAtInputsArrayIndexAndItsHeaderAndTriple(node, slot_index_in_inputs_array) {
  if (!node.inputs || slot_index_in_inputs_array < 0 || slot_index_in_inputs_array >= node.inputs.length) return;
  const input_descriptor = node.inputs[slot_index_in_inputs_array];
  const match = input_descriptor && input_descriptor.name
    ? input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX)
    : null;
  if (!match) return;
  const slot_number_string = match[1];

  node.removeInput(slot_index_in_inputs_array);

  removeWidgetsByNameOnNode(node, [
    `conditioning_${slot_number_string}_weight`,
    `conditioning_${slot_number_string}_end`,
    `conditioning_${slot_number_string}_start`,
    `__header_for_slot_${slot_number_string}`,
  ]);
}

// -------------- widget array layout normalize --------------

function rebuildWidgetsArrayWithHeadersInFrontOfEachSlotTripleOnNode(node) {
  // Preserve current values so the rebuild doesn't reset user-edited fields.
  const captured_widget_values_by_name = {};
  for (const widget_descriptor of node.widgets || []) {
    if (widget_descriptor && widget_descriptor.name && widget_descriptor.value !== undefined) {
      captured_widget_values_by_name[widget_descriptor.name] = widget_descriptor.value;
    }
  }
  // Wipe everything past the merge_mode widget at index 0.
  while (node.widgets && node.widgets.length > 1) {
    node.widgets.pop();
  }
  // Re-add header + triple for each conditioning_<N> input slot, restoring
  // captured values where we had them.
  for (const input_descriptor of node.inputs || []) {
    const match = input_descriptor && input_descriptor.name
      ? input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX)
      : null;
    if (!match) continue;
    const slot_number = parseInt(match[1], 10);
    addOneSlotHeaderLabelWidget(node, slot_number);
    addOneWidgetTripleForSlot(node, slot_number);
    for (const suffix of ["_start", "_end", "_weight"]) {
      const widget_name = `conditioning_${slot_number}${suffix}`;
      if (captured_widget_values_by_name[widget_name] !== undefined) {
        const widget_index = node.widgets.findIndex((w) => w && w.name === widget_name);
        if (widget_index >= 0) {
          node.widgets[widget_index].value = captured_widget_values_by_name[widget_name];
        }
      }
    }
  }
}

// -------------- stabilize: auto-expand on connect, auto-collapse on disconnect --------------

function stabilizeDynamicSlotsOnConditioningMergeNode(node) {
  if (!node.inputs) return;

  // Step 1: remove EVERY unconnected conditioning slot. This collapses middle
  // gaps AND removes the previous trailing empty. (We'll re-add a single
  // trailing empty at the end in step 2.)
  // Walk from the end so removeInput index math stays valid.
  for (let slot_index_in_inputs_array = node.inputs.length - 1; slot_index_in_inputs_array >= 0; slot_index_in_inputs_array--) {
    const input_descriptor = node.inputs[slot_index_in_inputs_array];
    if (!input_descriptor || !input_descriptor.name) continue;
    if (!CONDITIONING_INPUT_NAME_REGEX.test(input_descriptor.name)) continue;
    if (input_descriptor.link != null) continue; // connected — keep
    removeConditioningSlotAtInputsArrayIndexAndItsHeaderAndTriple(node, slot_index_in_inputs_array);
  }

  // Step 2: ensure exactly one trailing empty conditioning slot exists, with
  // a slot number one greater than the highest remaining (or 1 if none).
  const trailing_empty_index_after_collapse = indexOfTrailingEmptyConditioningSlotInInputsArrayOrMinusOne(node);
  if (trailing_empty_index_after_collapse < 0) {
    const next_slot_number = findHighestConditioningSlotIndexFromExistingInputs(node) + 1;
    addOneConditioningInputSlotAndItsWidgetTripleAndHeader(node, next_slot_number);
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

// -------------- extension registration --------------

app.registerExtension({
  name: "ConditioningMergeWithTimestepRanges.DynamicSlots",
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== NODE_TYPE_NAME_FOR_THIS_EXTENSION) return;

    const original_on_node_created_function = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      // Critical for widget value persistence across save/load.
      this.serialize_widgets = true;
      const original_result = original_on_node_created_function
        ? original_on_node_created_function.apply(this, arguments)
        : undefined;
      // ComfyUI's auto-build creates widgets for INPUT_TYPES.required
      // (merge_mode + slot 1 triple) without our header. Rebuild the widget
      // array so every slot's triple has its header above it.
      rebuildWidgetsArrayWithHeadersInFrontOfEachSlotTripleOnNode(this);
      scheduleStabilizationOnNode(this);
      return original_result;
    };

    const original_configure_function = nodeType.prototype.configure;
    nodeType.prototype.configure = function (serialized_node_data_object) {
      // Strip widgets past merge_mode and rebuild the right number of
      // headers + triples BEFORE delegating to original configure (which
      // positionally restores widgets_values).
      const restored_conditioning_slot_count_from_saved_inputs = (serialized_node_data_object && serialized_node_data_object.inputs || [])
        .filter((input_descriptor) =>
          input_descriptor && input_descriptor.name && CONDITIONING_INPUT_NAME_REGEX.test(input_descriptor.name)
        )
        .length;

      while (this.widgets && this.widgets.length > 1) {
        this.widgets.pop();
      }
      for (let slot_number_being_rebuilt = 1; slot_number_being_rebuilt <= restored_conditioning_slot_count_from_saved_inputs; slot_number_being_rebuilt++) {
        addOneSlotHeaderLabelWidget(this, slot_number_being_rebuilt);
        addOneWidgetTripleForSlot(this, slot_number_being_rebuilt);
      }

      const original_result = original_configure_function
        ? original_configure_function.apply(this, arguments)
        : undefined;
      scheduleStabilizationOnNode(this);
      return original_result;
    };

    const original_on_connections_change_function = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (slot_io_type, slot_index, is_connected, link_info, io_slot_descriptor) {
      const original_result = original_on_connections_change_function
        ? original_on_connections_change_function.apply(this, arguments)
        : undefined;
      if (slot_io_type === LITEGRAPH_INPUT_SLOT_IO_TYPE_ENUM_VALUE) {
        scheduleStabilizationOnNode(this);
      }
      return original_result;
    };
  },
});
