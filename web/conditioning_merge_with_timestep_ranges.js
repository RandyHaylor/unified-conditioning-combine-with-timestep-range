// Dynamic-slot frontend extension for ConditioningMergeWithTimestepRanges.
//
// - Maintains one trailing empty CONDITIONING input slot. Connecting it adds
//   a new empty slot (with its three widgets: _start, _end, _weight).
// - Per-slot canvas-drawn X button (right edge of the slot row): always
//   visible on slots index >= 1 (the required conditioning_1 has none).
//   Greyed out when the slot is connected (no-op on click; tooltip
//   "Disconnect to remove"); active when empty (click removes the slot
//   plus its widget triple).
// - Persists widget values across save/load:
//     * sets `serialize_widgets = true` on the node
//     * overrides `configure(info)` to rebuild the widget triples for the
//       restored slot count BEFORE LiteGraph populates widget values
//       (reference: rgthree-comfy/web/comfyui/power_lora_loader.js:21,51-70).
//
// Depends only on ComfyUI core (`../../scripts/app.js`) and LiteGraph core
// methods (`addInput` / `removeInput` / `addWidget`; `inputs[]` / `widgets[]`
// arrays; `setDirtyCanvas`). No third-party imports.

import { app } from "../../scripts/app.js";

const NODE_TYPE_NAME_FOR_THIS_EXTENSION = "ConditioningMergeWithTimestepRanges";

const STABILIZE_DEBOUNCE_DELAY_MILLISECONDS = 32;

const CONDITIONING_INPUT_NAME_REGEX = /^conditioning_(\d+)$/;

const REMOVE_BUTTON_X_ICON_SIZE_PIXELS = 12;
const REMOVE_BUTTON_X_ICON_RIGHT_MARGIN_PIXELS = 6;
const REMOVE_BUTTON_X_ICON_ACTIVE_ALPHA = 0.9;
const REMOVE_BUTTON_X_ICON_GREYED_ALPHA = 0.28;
const REMOVE_BUTTON_X_ICON_HOVER_HALO_ALPHA = 0.18;
const REMOVE_BUTTON_TOOLTIP_TEXT_FOR_EMPTY_REMOVABLE_SLOT = "Click to remove this input";
const REMOVE_BUTTON_TOOLTIP_TEXT_FOR_CONNECTED_SLOT = "Disconnect to remove";
const REMOVE_BUTTON_TOOLTIP_FONT_CSS = "11px Arial, sans-serif";

const LITEGRAPH_INPUT_SLOT_IO_TYPE_ENUM_VALUE = 1;

const REQUIRED_FIRST_SLOT_INDEX_IN_INPUTS_ARRAY = 0; // conditioning_1 is required and never gets an X

// -------------- helpers: structure / slot accounting --------------

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

function findLastConnectedConditioningSlotNumber(node) {
  let last_connected_slot_number = 0;
  for (const input_descriptor of node.inputs || []) {
    const match = input_descriptor && input_descriptor.name
      ? input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX)
      : null;
    if (!match) continue;
    if (input_descriptor.link != null) {
      const slot_number = parseInt(match[1], 10);
      if (slot_number > last_connected_slot_number) last_connected_slot_number = slot_number;
    }
  }
  return last_connected_slot_number;
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

function indexOfTrailingEmptyConditioningSlotInInputsArrayOrMinusOne(node) {
  if (!node.inputs || node.inputs.length === 0) return -1;
  const last_index_in_inputs_array = node.inputs.length - 1;
  const last_input_descriptor = node.inputs[last_index_in_inputs_array];
  if (!last_input_descriptor || !last_input_descriptor.name) return -1;
  if (!CONDITIONING_INPUT_NAME_REGEX.test(last_input_descriptor.name)) return -1;
  if (last_input_descriptor.link != null) return -1;
  return last_index_in_inputs_array;
}

// -------------- helpers: add / remove slot + widgets --------------

function addOneConditioningInputSlotAndItsWidgetTriple(node, slot_number_to_use) {
  node.addInput(`conditioning_${slot_number_to_use}`, "CONDITIONING");
  addOneWidgetTripleForSlot(node, slot_number_to_use);
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

function removeOneConditioningInputSlotAtIndexAndItsWidgetTriple(node, slot_index_in_inputs_array) {
  if (!node.inputs || slot_index_in_inputs_array < 0 || slot_index_in_inputs_array >= node.inputs.length) return;
  const input_descriptor = node.inputs[slot_index_in_inputs_array];
  const match = input_descriptor && input_descriptor.name
    ? input_descriptor.name.match(CONDITIONING_INPUT_NAME_REGEX)
    : null;
  if (!match) return;
  const slot_number_string_extracted_from_input_name = match[1];

  node.removeInput(slot_index_in_inputs_array);

  const widget_name_suffixes_to_remove_in_reverse_order = ["_weight", "_end", "_start"];
  for (const suffix of widget_name_suffixes_to_remove_in_reverse_order) {
    const widget_name_to_find = `conditioning_${slot_number_string_extracted_from_input_name}${suffix}`;
    const widget_index_in_widgets_array = (node.widgets || []).findIndex(
      (widget_descriptor) => widget_descriptor && widget_descriptor.name === widget_name_to_find,
    );
    if (widget_index_in_widgets_array >= 0) {
      node.widgets.splice(widget_index_in_widgets_array, 1);
    }
  }
}

// -------------- stabilize: maintain trailing empty --------------

function stabilizeDynamicSlotsOnConditioningMergeNode(node) {
  const total_conditioning_slots_currently = countConditioningInputSlotsOnNode(node);
  const last_connected_slot_number = findLastConnectedConditioningSlotNumber(node);
  const desired_total_conditioning_slots = Math.max(1, last_connected_slot_number + 1);

  // Remove trailing UNCONNECTED slots beyond desired count.
  while (countConditioningInputSlotsOnNode(node) > desired_total_conditioning_slots) {
    const last_index_in_inputs_array = node.inputs.length - 1;
    const last_input_descriptor = node.inputs[last_index_in_inputs_array];
    const is_conditioning_slot = last_input_descriptor
      && last_input_descriptor.name
      && CONDITIONING_INPUT_NAME_REGEX.test(last_input_descriptor.name);
    const is_unconnected = !last_input_descriptor.link;
    if (is_conditioning_slot && is_unconnected) {
      removeOneConditioningInputSlotAtIndexAndItsWidgetTriple(node, last_index_in_inputs_array);
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

// -------------- X-button geometry / hit test / draw --------------

function getRemoveButtonXIconBoundsForSlotIndexOrNull(node, slot_index_in_inputs_array) {
  // Skip slot 0 (required conditioning_1) and skip the trailing-empty slot.
  if (slot_index_in_inputs_array <= REQUIRED_FIRST_SLOT_INDEX_IN_INPUTS_ARRAY) return null;
  const input_descriptor = node.inputs && node.inputs[slot_index_in_inputs_array];
  if (!input_descriptor || !input_descriptor.name) return null;
  if (!CONDITIONING_INPUT_NAME_REGEX.test(input_descriptor.name)) return null;

  const trailing_empty_index = indexOfTrailingEmptyConditioningSlotInInputsArrayOrMinusOne(node);
  if (slot_index_in_inputs_array === trailing_empty_index) return null;

  // Slot vertical center in node-local coordinates.
  const SLOT_HEIGHT = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_SLOT_HEIGHT) || 20;
  const TITLE_HEIGHT = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_TITLE_HEIGHT) || 24;
  const slot_y_center = TITLE_HEIGHT + SLOT_HEIGHT * (slot_index_in_inputs_array + 0.5);

  const icon_top_left_x = node.size[0] - REMOVE_BUTTON_X_ICON_RIGHT_MARGIN_PIXELS - REMOVE_BUTTON_X_ICON_SIZE_PIXELS;
  const icon_top_left_y = slot_y_center - REMOVE_BUTTON_X_ICON_SIZE_PIXELS / 2;
  return {
    x: icon_top_left_x,
    y: icon_top_left_y,
    width: REMOVE_BUTTON_X_ICON_SIZE_PIXELS,
    height: REMOVE_BUTTON_X_ICON_SIZE_PIXELS,
  };
}

function pointIsInsideBoundingRectangle(point_x_in_node_local_coords, point_y_in_node_local_coords, bounding_rectangle) {
  return (
    point_x_in_node_local_coords >= bounding_rectangle.x
    && point_x_in_node_local_coords <= bounding_rectangle.x + bounding_rectangle.width
    && point_y_in_node_local_coords >= bounding_rectangle.y
    && point_y_in_node_local_coords <= bounding_rectangle.y + bounding_rectangle.height
  );
}

function findSlotIndexWhereRemoveButtonContainsNodeLocalPointOrMinusOne(node, point_x_in_node_local_coords, point_y_in_node_local_coords) {
  if (!node.inputs) return -1;
  for (let slot_index_in_inputs_array = 0; slot_index_in_inputs_array < node.inputs.length; slot_index_in_inputs_array++) {
    const bounds = getRemoveButtonXIconBoundsForSlotIndexOrNull(node, slot_index_in_inputs_array);
    if (bounds && pointIsInsideBoundingRectangle(point_x_in_node_local_coords, point_y_in_node_local_coords, bounds)) {
      return slot_index_in_inputs_array;
    }
  }
  return -1;
}

function drawRemoveButtonXIconsOnAllApplicableSlots(node, canvas_context) {
  if (!node.inputs) return;
  const hovered_slot_index = (node.__conditioning_merge_hovered_x_icon_slot_index == null)
    ? -1
    : node.__conditioning_merge_hovered_x_icon_slot_index;

  canvas_context.save();
  for (let slot_index_in_inputs_array = 0; slot_index_in_inputs_array < node.inputs.length; slot_index_in_inputs_array++) {
    const bounds = getRemoveButtonXIconBoundsForSlotIndexOrNull(node, slot_index_in_inputs_array);
    if (!bounds) continue;

    const input_descriptor = node.inputs[slot_index_in_inputs_array];
    const slot_is_connected = input_descriptor.link != null;
    const alpha_for_this_x_icon = slot_is_connected
      ? REMOVE_BUTTON_X_ICON_GREYED_ALPHA
      : REMOVE_BUTTON_X_ICON_ACTIVE_ALPHA;

    // Optional hover halo
    if (slot_index_in_inputs_array === hovered_slot_index) {
      canvas_context.globalAlpha = REMOVE_BUTTON_X_ICON_HOVER_HALO_ALPHA;
      canvas_context.fillStyle = "#ffffff";
      canvas_context.beginPath();
      canvas_context.arc(
        bounds.x + bounds.width / 2,
        bounds.y + bounds.height / 2,
        bounds.width * 0.75,
        0,
        Math.PI * 2,
      );
      canvas_context.fill();
    }

    canvas_context.globalAlpha = alpha_for_this_x_icon;
    canvas_context.strokeStyle = "#ffffff";
    canvas_context.lineWidth = 1.5;
    canvas_context.lineCap = "round";
    const padding_inside_icon_box = 2.5;
    canvas_context.beginPath();
    canvas_context.moveTo(bounds.x + padding_inside_icon_box, bounds.y + padding_inside_icon_box);
    canvas_context.lineTo(bounds.x + bounds.width - padding_inside_icon_box, bounds.y + bounds.height - padding_inside_icon_box);
    canvas_context.moveTo(bounds.x + bounds.width - padding_inside_icon_box, bounds.y + padding_inside_icon_box);
    canvas_context.lineTo(bounds.x + padding_inside_icon_box, bounds.y + bounds.height - padding_inside_icon_box);
    canvas_context.stroke();
  }
  canvas_context.restore();
}

function drawTooltipIfHoveringAnyRemoveButtonXIcon(node, canvas_context) {
  const hovered_slot_index = node.__conditioning_merge_hovered_x_icon_slot_index;
  if (hovered_slot_index == null || hovered_slot_index < 0) return;
  if (!node.inputs || hovered_slot_index >= node.inputs.length) return;
  const input_descriptor = node.inputs[hovered_slot_index];
  if (!input_descriptor) return;
  const bounds = getRemoveButtonXIconBoundsForSlotIndexOrNull(node, hovered_slot_index);
  if (!bounds) return;

  const slot_is_connected = input_descriptor.link != null;
  const tooltip_text = slot_is_connected
    ? REMOVE_BUTTON_TOOLTIP_TEXT_FOR_CONNECTED_SLOT
    : REMOVE_BUTTON_TOOLTIP_TEXT_FOR_EMPTY_REMOVABLE_SLOT;

  canvas_context.save();
  canvas_context.font = REMOVE_BUTTON_TOOLTIP_FONT_CSS;
  const measured_text_width = canvas_context.measureText(tooltip_text).width;
  const tooltip_box_padding_x = 6;
  const tooltip_box_padding_y = 4;
  const tooltip_box_width = measured_text_width + tooltip_box_padding_x * 2;
  const tooltip_box_height = 18;
  // Position to the left of the X so it stays inside the node.
  const tooltip_box_x = bounds.x - tooltip_box_width - 6;
  const tooltip_box_y = bounds.y + bounds.height / 2 - tooltip_box_height / 2;

  canvas_context.fillStyle = "rgba(20,20,20,0.94)";
  canvas_context.fillRect(tooltip_box_x, tooltip_box_y, tooltip_box_width, tooltip_box_height);
  canvas_context.strokeStyle = "rgba(255,255,255,0.25)";
  canvas_context.lineWidth = 1;
  canvas_context.strokeRect(tooltip_box_x + 0.5, tooltip_box_y + 0.5, tooltip_box_width - 1, tooltip_box_height - 1);

  canvas_context.fillStyle = "#ffffff";
  canvas_context.textBaseline = "middle";
  canvas_context.fillText(tooltip_text, tooltip_box_x + tooltip_box_padding_x, tooltip_box_y + tooltip_box_height / 2);
  canvas_context.restore();
}

// -------------- click / hover handlers --------------

function handleClickOnRemoveButtonXIconReturningWhetherClickWasConsumed(node, point_x_in_node_local_coords, point_y_in_node_local_coords) {
  const slot_index_clicked = findSlotIndexWhereRemoveButtonContainsNodeLocalPointOrMinusOne(
    node, point_x_in_node_local_coords, point_y_in_node_local_coords
  );
  if (slot_index_clicked < 0) return false;
  const input_descriptor = node.inputs[slot_index_clicked];
  if (!input_descriptor) return false;
  const slot_is_connected = input_descriptor.link != null;
  if (slot_is_connected) {
    // No-op when connected; click consumed so it doesn't fall through.
    return true;
  }
  removeOneConditioningInputSlotAtIndexAndItsWidgetTriple(node, slot_index_clicked);
  scheduleStabilizationOnNode(node);
  return true;
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

    // Override configure to rebuild widgets matching the restored slot count
    // BEFORE delegating to the original configure (which populates widget
    // values from info.widgets_values positionally).
    const original_configure_function = nodeType.prototype.configure;
    nodeType.prototype.configure = function (serialized_node_data_object) {
      const restored_conditioning_slot_count_from_saved_inputs = (serialized_node_data_object && serialized_node_data_object.inputs || [])
        .filter((input_descriptor) =>
          input_descriptor && input_descriptor.name && CONDITIONING_INPUT_NAME_REGEX.test(input_descriptor.name)
        )
        .length;

      // Strip all widgets except the merge_mode dropdown at index 0.
      while (this.widgets && this.widgets.length > 1) {
        this.widgets.pop();
      }
      // Rebuild widget triples for each restored slot.
      for (let slot_number_being_rebuilt = 1; slot_number_being_rebuilt <= restored_conditioning_slot_count_from_saved_inputs; slot_number_being_rebuilt++) {
        addOneWidgetTripleForSlot(this, slot_number_being_rebuilt);
      }

      const original_result = original_configure_function
        ? original_configure_function.apply(this, arguments)
        : undefined;
      // Post-restore: ensure trailing empty exists.
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

    const original_on_draw_foreground_function = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function (canvas_context) {
      const original_result = original_on_draw_foreground_function
        ? original_on_draw_foreground_function.apply(this, arguments)
        : undefined;
      // Skip drawing when collapsed.
      if (this.flags && this.flags.collapsed) return original_result;
      drawRemoveButtonXIconsOnAllApplicableSlots(this, canvas_context);
      drawTooltipIfHoveringAnyRemoveButtonXIcon(this, canvas_context);
      return original_result;
    };

    const original_on_mouse_down_function = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function (mouse_event, point_in_node_local_coords, graph_canvas) {
      const point_x_in_node_local_coords = point_in_node_local_coords[0];
      const point_y_in_node_local_coords = point_in_node_local_coords[1];
      const click_was_consumed = handleClickOnRemoveButtonXIconReturningWhetherClickWasConsumed(
        this, point_x_in_node_local_coords, point_y_in_node_local_coords
      );
      if (click_was_consumed) return true;
      return original_on_mouse_down_function
        ? original_on_mouse_down_function.apply(this, arguments)
        : undefined;
    };

    const original_on_mouse_move_function = nodeType.prototype.onMouseMove;
    nodeType.prototype.onMouseMove = function (mouse_event, point_in_node_local_coords, graph_canvas) {
      const point_x_in_node_local_coords = point_in_node_local_coords[0];
      const point_y_in_node_local_coords = point_in_node_local_coords[1];
      const newly_hovered_slot_index = findSlotIndexWhereRemoveButtonContainsNodeLocalPointOrMinusOne(
        this, point_x_in_node_local_coords, point_y_in_node_local_coords
      );
      if (newly_hovered_slot_index !== this.__conditioning_merge_hovered_x_icon_slot_index) {
        this.__conditioning_merge_hovered_x_icon_slot_index = newly_hovered_slot_index;
        this.setDirtyCanvas(true);
      }
      return original_on_mouse_move_function
        ? original_on_mouse_move_function.apply(this, arguments)
        : undefined;
    };

    // Clear hover state when mouse leaves the node.
    const original_on_mouse_leave_function = nodeType.prototype.onMouseLeave;
    nodeType.prototype.onMouseLeave = function (mouse_event, point_in_node_local_coords, graph_canvas) {
      if (this.__conditioning_merge_hovered_x_icon_slot_index != null && this.__conditioning_merge_hovered_x_icon_slot_index >= 0) {
        this.__conditioning_merge_hovered_x_icon_slot_index = -1;
        this.setDirtyCanvas(true);
      }
      return original_on_mouse_leave_function
        ? original_on_mouse_leave_function.apply(this, arguments)
        : undefined;
    };
  },
});
