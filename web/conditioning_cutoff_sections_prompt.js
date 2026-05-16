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

const SECTION_WIDGET_NAME_REGEX = /^section_(\d+)_(text|isolate)$/;

const HIDDEN_WIDGET_COMPUTE_SIZE_RETURN_HEIGHT = -4;

function setVisibilityOfOneWidget(widget_to_show_or_hide, should_be_hidden) {
  if (!widget_to_show_or_hide) return;
  if (should_be_hidden) {
    if (widget_to_show_or_hide.__cutoff_original_compute_size === undefined) {
      widget_to_show_or_hide.__cutoff_original_compute_size = widget_to_show_or_hide.computeSize;
      widget_to_show_or_hide.__cutoff_original_type = widget_to_show_or_hide.type;
    }
    widget_to_show_or_hide.computeSize = function () {
      return [0, HIDDEN_WIDGET_COMPUTE_SIZE_RETURN_HEIGHT];
    };
    widget_to_show_or_hide.type = "hidden";
  } else {
    if (widget_to_show_or_hide.__cutoff_original_compute_size !== undefined) {
      widget_to_show_or_hide.computeSize = widget_to_show_or_hide.__cutoff_original_compute_size;
      widget_to_show_or_hide.type = widget_to_show_or_hide.__cutoff_original_type;
      widget_to_show_or_hide.__cutoff_original_compute_size = undefined;
      widget_to_show_or_hide.__cutoff_original_type = undefined;
    }
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
  // Resize the node to fit the new widget layout.
  if (typeof node.computeSize === "function") {
    const min_size_for_visible_widgets = node.computeSize();
    // Preserve current width; bump height to new minimum.
    node.size[1] = min_size_for_visible_widgets[1];
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
  },
});
