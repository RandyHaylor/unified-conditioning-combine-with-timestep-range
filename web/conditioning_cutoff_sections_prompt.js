// Frontend extension for ConditioningCutoffSectionsPrompt.
//
// Hides section_N_text / section_N_isolate widget pairs whose index N is
// greater than the current `section_count` widget value. The hidden widgets
// stay in `node.widgets[]` (so their values still serialize to / restore
// from widgets_values), they just render at zero height.
//
// On change of section_count, on node creation, and on configure (workflow
// load), visibility is recomputed.
//
// Depends only on ComfyUI core (`../../scripts/app.js`) and LiteGraph core
// methods (`computeSize` override, `setDirtyCanvas`).

import { app } from "../../scripts/app.js";

const NODE_TYPE_NAME_FOR_THIS_EXTENSION = "ConditioningCutoffSectionsPrompt";

const SECTION_WIDGET_NAME_REGEX = /^section_(\d+)_(text|isolate|weight)$/;

const HIDDEN_WIDGET_COMPUTE_SIZE_RETURN_HEIGHT = -4;

// ComfyUI's frontend treats widgets whose type starts with "converted-widget"
// as the special "convert-to-input" hidden state, which is the canonical
// way to make BOTH the canvas widget AND its DOM companion (textarea / input
// element for STRING multiline / FLOAT widgets) disappear. Setting
// `type = "hidden"` alone is NOT recognized by ComfyUI's renderer and lets
// the DOM element keep rendering off-position.
const CONVERTED_HIDDEN_WIDGET_TYPE_SENTINEL = "converted-widget";

function setVisibilityOfOneWidget(widget_to_show_or_hide, should_be_hidden) {
  if (!widget_to_show_or_hide) return;
  // Two things to hide / show:
  //   (a) the canvas-drawn widget (controlled via computeSize + type)
  //   (b) the DOM element ComfyUI may have inserted for STRING multiline /
  //       FLOAT / etc. widgets — these are positioned outside the canvas
  //       and ignore the computeSize trick, so they'd "hang off" the node
  //       bottom when hidden.
  if (should_be_hidden) {
    if (widget_to_show_or_hide.__cutoff_original_compute_size === undefined) {
      widget_to_show_or_hide.__cutoff_original_compute_size = widget_to_show_or_hide.computeSize;
      widget_to_show_or_hide.__cutoff_original_type = widget_to_show_or_hide.type;
      widget_to_show_or_hide.__cutoff_original_serialize_value_function = widget_to_show_or_hide.serializeValue;
    }
    widget_to_show_or_hide.computeSize = function () {
      return [0, HIDDEN_WIDGET_COMPUTE_SIZE_RETURN_HEIGHT];
    };
    // Sentinel type telling ComfyUI's renderer to skip drawing AND to hide
    // the DOM companion (textarea / input). Append a per-widget suffix so
    // each hidden widget is distinguishable in the renderer's bookkeeping,
    // matching the comfy-mtb hideWidget pattern.
    widget_to_show_or_hide.type = CONVERTED_HIDDEN_WIDGET_TYPE_SENTINEL + ":" + widget_to_show_or_hide.name;
    widget_to_show_or_hide.hidden = true;
    // Preserve the underlying value during hide via a serializeValue that
    // returns the value-as-was; this keeps the value in widgets_values on
    // save so re-expand picks up the same text.
    const original_serialize_function_at_hide_time = widget_to_show_or_hide.__cutoff_original_serialize_value_function;
    widget_to_show_or_hide.serializeValue = function () {
      if (original_serialize_function_at_hide_time) {
        return original_serialize_function_at_hide_time.apply(this, arguments);
      }
      return widget_to_show_or_hide.value;
    };
  } else {
    if (widget_to_show_or_hide.__cutoff_original_compute_size !== undefined) {
      widget_to_show_or_hide.computeSize = widget_to_show_or_hide.__cutoff_original_compute_size;
      widget_to_show_or_hide.type = widget_to_show_or_hide.__cutoff_original_type;
      if (widget_to_show_or_hide.__cutoff_original_serialize_value_function !== undefined) {
        widget_to_show_or_hide.serializeValue = widget_to_show_or_hide.__cutoff_original_serialize_value_function;
      }
      widget_to_show_or_hide.__cutoff_original_compute_size = undefined;
      widget_to_show_or_hide.__cutoff_original_type = undefined;
      widget_to_show_or_hide.__cutoff_original_serialize_value_function = undefined;
    }
    widget_to_show_or_hide.hidden = false;
  }
}

function findWidgetByNameOnNodeOrUndefined(node, widget_name_to_find) {
  if (!node.widgets) return undefined;
  for (const widget_descriptor of node.widgets) {
    if (widget_descriptor && widget_descriptor.name === widget_name_to_find) {
      return widget_descriptor;
    }
  }
  return undefined;
}

function updateAllSectionWidgetVisibilityBasedOnCurrentSectionCountValue(node) {
  const section_count_widget = findWidgetByNameOnNodeOrUndefined(node, "section_count");
  if (!section_count_widget) return;
  const current_section_count_value = Math.max(
    1, Math.floor(Number(section_count_widget.value) || 1)
  );
  for (const widget_descriptor of node.widgets || []) {
    if (!widget_descriptor || !widget_descriptor.name) continue;
    const regex_match_for_section_widget_name = widget_descriptor.name.match(SECTION_WIDGET_NAME_REGEX);
    if (!regex_match_for_section_widget_name) continue;
    const this_widget_section_index = parseInt(regex_match_for_section_widget_name[1], 10);
    setVisibilityOfOneWidget(widget_descriptor, this_widget_section_index > current_section_count_value);
  }
  // Resize the node to fit the new widget layout. Use the canonical
  // setSize([max_width, new_height]) pattern (matches comfy-mtb's
  // convertToWidget at comfy_shared.js:171) so LiteGraph re-lays-out
  // widgets correctly — direct mutation of node.size[1] doesn't
  // trigger the relayout that newly-visible widgets need.
  if (typeof node.computeSize === "function" && typeof node.setSize === "function") {
    const min_size_for_currently_visible_widgets = node.computeSize();
    const new_width_preserving_user_drag = Math.max(node.size[0], min_size_for_currently_visible_widgets[0]);
    const new_height_to_fit_visible_widgets = min_size_for_currently_visible_widgets[1];
    node.setSize([new_width_preserving_user_drag, new_height_to_fit_visible_widgets]);
  }
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "ConditioningCutoffSectionsPrompt.SectionVisibility",
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== NODE_TYPE_NAME_FOR_THIS_EXTENSION) return;

    const original_on_node_created_function = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      this.serialize_widgets = true;
      const original_on_node_created_return_value = original_on_node_created_function
        ? original_on_node_created_function.apply(this, arguments)
        : undefined;

      // Wrap the section_count widget's callback so visibility updates
      // whenever the value changes.
      const section_count_widget = findWidgetByNameOnNodeOrUndefined(this, "section_count");
      if (section_count_widget) {
        const previous_section_count_widget_callback = section_count_widget.callback;
        const node_reference_for_callback = this;
        section_count_widget.callback = function () {
          const previous_callback_return_value = previous_section_count_widget_callback
            ? previous_section_count_widget_callback.apply(this, arguments)
            : undefined;
          updateAllSectionWidgetVisibilityBasedOnCurrentSectionCountValue(node_reference_for_callback);
          return previous_callback_return_value;
        };
      }

      updateAllSectionWidgetVisibilityBasedOnCurrentSectionCountValue(this);
      // DOM elements for multiline STRING / FLOAT widgets are sometimes
      // inserted asynchronously after onNodeCreated returns. Run a second
      // pass next tick so the DOM hide also gets applied.
      const node_reference_for_deferred_update = this;
      setTimeout(function () {
        updateAllSectionWidgetVisibilityBasedOnCurrentSectionCountValue(node_reference_for_deferred_update);
      }, 0);
      return original_on_node_created_return_value;
    };

    const original_on_configure_function = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (_serialized_node_data) {
      const original_on_configure_return_value = original_on_configure_function
        ? original_on_configure_function.apply(this, arguments)
        : undefined;
      // After workflow load, refresh visibility from the restored
      // section_count widget value.
      updateAllSectionWidgetVisibilityBasedOnCurrentSectionCountValue(this);
      return original_on_configure_return_value;
    };

    // ComfyUI fires node.onWidgetChanged when any widget's value changes.
    // Hooking it as a second path (in addition to wrapping section_count's
    // callback) makes the update fire reliably even if the callback wrap
    // is bypassed by some widget implementations.
    const original_on_widget_changed_function = nodeType.prototype.onWidgetChanged;
    nodeType.prototype.onWidgetChanged = function (changed_widget_name, _new_widget_value, _old_widget_value, _changed_widget) {
      const original_on_widget_changed_return_value = original_on_widget_changed_function
        ? original_on_widget_changed_function.apply(this, arguments)
        : undefined;
      if (changed_widget_name === "section_count") {
        updateAllSectionWidgetVisibilityBasedOnCurrentSectionCountValue(this);
      }
      return original_on_widget_changed_return_value;
    };
  },
});
