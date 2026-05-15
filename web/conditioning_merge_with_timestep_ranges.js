// Dynamic-slot frontend extension for ConditioningMergeWithTimestepRanges.
//
// Behavior:
//   - Each CONNECTED conditioning_<N> slot gets a small header label
//     "── slot N ──" plus three widgets (start, end, weight).
//   - An empty / unconnected slot has NO header and NO widgets — it's just
//     an input pin waiting for a connection. Exactly one trailing empty slot
//     is always maintained.
//   - Connecting the trailing empty slot causes its widgets to appear and a
//     new (empty, widgetless) trailing slot to be added.
//   - Disconnecting any slot removes that slot (auto-collapse). Its widget
//     values are not preserved across the disconnect — reconnecting starts
//     at defaults.
//   - Widget values persist across save/load:
//       * `serialize_widgets = true` is set on the node.
//       * `configure(info)` is overridden to rebuild widget triples for every
//         CONNECTED slot before LiteGraph positionally restores widget
//         values from info.widgets_values.
//
// Depends only on ComfyUI core (`../../scripts/app.js`) and LiteGraph core
// methods (addInput / removeInput / setDirtyCanvas, plus the inputs[] /
// widgets[] arrays). No third-party imports.

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

// -------------- widget add helpers --------------

function addOneSlotHeaderLabelWidget(node, slot_number_to_use) {
  // Header widget that visually groups its slot's triple. It DOES serialize
  // (value="") so save and load both walk widgets_values in the same order
  // — using options.serialize=false caused positional misalignment on load
  // (slot 1's start widget would end up assigned the saved end value).
  // The displayed slot number is read dynamically from a property on the
  // widget instance so a rename pass can update it without recreating.
  const slot_header_label_widget = {
    name: `__header_for_slot_${slot_number_to_use}`,
    type: "header",
    value: "",
    slot_number_for_display_only: slot_number_to_use,
    draw(canvas_context, owning_node, widget_width_pixels, y_top_pixels, widget_height_pixels) {
      canvas_context.save();
      canvas_context.fillStyle = "#9aa";
      canvas_context.font = "bold 11px Arial, sans-serif";
      canvas_context.textBaseline = "middle";
      const label_text = `── slot ${this.slot_number_for_display_only} ──`;
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

const CLIP_STREAM_PASS_CHOICES_IN_DROPDOWN_ORDER = ["Pass L+G", "Pass L", "Pass G"];
const CLIP_STREAM_PASS_DEFAULT_VALUE = "Pass L+G";

function addOneWidgetTripleForSlot(node, slot_number_to_use) {
  // Despite the historical "Triple" name, this now adds FOUR widgets per slot:
  // start, end, weight, and clip.
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
  node.addWidget(
    "combo",
    `conditioning_${slot_number_to_use}_clip`,
    CLIP_STREAM_PASS_DEFAULT_VALUE,
    () => {},
    { values: CLIP_STREAM_PASS_CHOICES_IN_DROPDOWN_ORDER },
  );
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

// -------------- widget array rebuild: ONE widget set per CONNECTED slot --------------

function rebuildWidgetsArrayMatchingCurrentlyConnectedSlotsOnNode(node) {
  // Preserve current widget values so a rebuild doesn't reset user-edited
  // fields for slots that are still connected.
  const captured_widget_values_by_name = {};
  for (const widget_descriptor of node.widgets || []) {
    if (widget_descriptor && widget_descriptor.name && widget_descriptor.value !== undefined) {
      captured_widget_values_by_name[widget_descriptor.name] = widget_descriptor.value;
    }
  }
  // Strip everything past the merge_mode widget at index 0.
  while (node.widgets && node.widgets.length > 1) {
    node.widgets.pop();
  }
  // Re-add header + triple ONLY for connected conditioning slots.
  for (const input_descriptor of node.inputs || []) {
    if (!input_descriptor || !input_descriptor.name) continue;
    const match = input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX);
    if (!match) continue;
    if (input_descriptor.link == null) continue; // skip empties
    const slot_number = parseInt(match[1], 10);
    addOneSlotHeaderLabelWidget(node, slot_number);
    addOneWidgetTripleForSlot(node, slot_number);
    // Restore captured values where available.
    for (const suffix of ["_start", "_end", "_weight", "_clip"]) {
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

function renumberRemainingConditioningSlotsContiguouslyStartingAtOne(node) {
  // Walks all conditioning slots in their current input-array order and
  // re-numbers them 1..N contiguously. Renames the input.name, the three
  // widgets per slot, and updates the header widget's displayed number.
  // LiteGraph links reference slot INDEX (not name), so renaming names is
  // safe for existing wires.
  if (!node.inputs) return;
  let next_desired_slot_number = 1;
  for (const input_descriptor of node.inputs) {
    if (!input_descriptor || !input_descriptor.name) continue;
    const match = input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX);
    if (!match) continue;
    const current_slot_number = parseInt(match[1], 10);
    if (current_slot_number !== next_desired_slot_number) {
      input_descriptor.name = `conditioning_${next_desired_slot_number}`;
      // Rename widgets.
      for (const suffix of ["_start", "_end", "_weight", "_clip"]) {
        const old_widget_name = `conditioning_${current_slot_number}${suffix}`;
        const new_widget_name = `conditioning_${next_desired_slot_number}${suffix}`;
        const widget_index = (node.widgets || []).findIndex((w) => w && w.name === old_widget_name);
        if (widget_index >= 0) {
          node.widgets[widget_index].name = new_widget_name;
        }
      }
      // Update header (rename + update displayed number).
      const old_header_name = `__header_for_slot_${current_slot_number}`;
      const new_header_name = `__header_for_slot_${next_desired_slot_number}`;
      const header_index = (node.widgets || []).findIndex((w) => w && w.name === old_header_name);
      if (header_index >= 0) {
        node.widgets[header_index].name = new_header_name;
        node.widgets[header_index].slot_number_for_display_only = next_desired_slot_number;
      }
    }
    next_desired_slot_number++;
  }
}

function stabilizeDynamicSlotsOnConditioningMergeNode(node) {
  if (!node.inputs) return;

  // Step 1: remove EVERY unconnected conditioning slot (collapses middle
  // gaps and the previous trailing empty). Walk from end so removeInput
  // index math stays valid.
  for (let slot_index_in_inputs_array = node.inputs.length - 1; slot_index_in_inputs_array >= 0; slot_index_in_inputs_array--) {
    const input_descriptor = node.inputs[slot_index_in_inputs_array];
    if (!input_descriptor || !input_descriptor.name) continue;
    if (!CONDITIONING_INPUT_NAME_REGEX.test(input_descriptor.name)) continue;
    if (input_descriptor.link != null) continue;
    node.removeInput(slot_index_in_inputs_array);
  }

  // Step 2: rebuild widgets so only currently-connected slots have a
  // header + triple. This also picks up newly-connected slots that didn't
  // have widgets before.
  rebuildWidgetsArrayMatchingCurrentlyConnectedSlotsOnNode(node);

  // Step 3: re-number connected slots so they are contiguous 1..N (no gaps).
  renumberRemainingConditioningSlotsContiguouslyStartingAtOne(node);

  // Step 4: add a trailing empty input slot WITHOUT widgets, numbered
  // one past the highest connected slot (post-renumbering this is N+1).
  const next_slot_number = findHighestConditioningSlotIndexFromExistingInputs(node) + 1;
  node.addInput(`conditioning_${next_slot_number}`, "CONDITIONING");

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
      scheduleStabilizationOnNode(this);
      return original_result;
    };

    const original_configure_function = nodeType.prototype.configure;
    nodeType.prototype.configure = function (serialized_node_data_object) {
      // Strip widgets past merge_mode, then add a header + triple for every
      // CONNECTED conditioning slot in the saved input list. Empty slots
      // (including the trailing empty) get no widgets.
      const saved_inputs_list = (serialized_node_data_object && serialized_node_data_object.inputs) || [];

      while (this.widgets && this.widgets.length > 1) {
        this.widgets.pop();
      }
      for (const input_descriptor of saved_inputs_list) {
        if (!input_descriptor || !input_descriptor.name) continue;
        const match = input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX);
        if (!match) continue;
        if (input_descriptor.link == null) continue;
        const slot_number_being_rebuilt = parseInt(match[1], 10);
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
