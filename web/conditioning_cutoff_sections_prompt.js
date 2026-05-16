// Frontend extension for ConditioningCutoffSectionsPrompt.
//
// Hides section_N_text / section_N_isolate / section_N_weight widget triples
// whose index N is greater than the current `section_count` widget value.
// The hidden widgets stay in `node.widgets[]` (so their values still serialize
// to / restore from widgets_values), they just render at zero height.
//
// Pattern adopted from comfyui-easy-use's toggleWidget at
//   custom_nodes/comfyui-easy-use/web_version/v1/js/common/utils.js:89-103
// Key bits:
//   - Cache widget.type + widget.computeSize on first hide so they can be
//     restored on show.
//   - Capture node.size BEFORE applying the toggle.
//   - On show: height = max(node.computeSize()[1], origSize[1])  -> grows.
//     On hide: height = node.size[1]                              -> stable.
//   - Always call node.setSize([node.size[0], height]) inside the toggle.
//   - After all toggles for one update pass, call updateNodeHeight once.

import { app } from "../../scripts/app.js";

const NODE_TYPE_NAME_FOR_THIS_EXTENSION = "ConditioningCutoffSectionsPrompt";

const SECTION_WIDGET_NAME_REGEX = /^section_(\d+)_(text|isolate|weight)$/;

const HIDDEN_WIDGET_TYPE_SENTINEL_PREFIX = "cutoffSectionsHidden:";

// One global cache keyed by widget.name. The widget objects themselves get
// re-created on workflow load, so storing the originals on the widget would
// not survive a configure pass.
const original_widget_props_cache_by_widget_name = {};

function findWidgetByNameOnNodeOrUndefined(node, widget_name_to_find) {
  if (!node.widgets) return undefined;
  for (const widget_descriptor of node.widgets) {
    if (widget_descriptor && widget_descriptor.name === widget_name_to_find) {
      return widget_descriptor;
    }
  }
  return undefined;
}

function toggleVisibilityOfOneWidgetOnNodeMatchingComfyuiEasyUsePattern(node, widget_to_toggle, should_be_visible) {
  if (!widget_to_toggle) return;

  // Cache the original type + computeSize ONCE per widget name.
  if (!original_widget_props_cache_by_widget_name[widget_to_toggle.name]) {
    original_widget_props_cache_by_widget_name[widget_to_toggle.name] = {
      original_type: widget_to_toggle.type,
      original_compute_size_function: widget_to_toggle.computeSize,
    };
  }
  const cached_original_props_for_this_widget = original_widget_props_cache_by_widget_name[widget_to_toggle.name];

  const node_size_captured_before_this_toggle = [node.size[0], node.size[1]];

  if (should_be_visible) {
    widget_to_toggle.type = cached_original_props_for_this_widget.original_type;
    // CRITICAL: do NOT restore widget.computeSize from cache. ComfyUI
    // computes widget sizes lazily based on widget.type via class-level
    // defaults; the cached value at first-hide may have been undefined
    // (verified empirically — after restore, widget.computeSize was
    // undefined, which made the widget render at sliver height).
    // `delete` the instance-level override so LiteGraph's default
    // computeSize for the restored type takes effect again.
    delete widget_to_toggle.computeSize;
    widget_to_toggle.hidden = false;
    // Also unhide any DOM companion (textarea for STRING multiline, etc.)
    if (widget_to_toggle.element && widget_to_toggle.element.style) {
      widget_to_toggle.element.style.display = "";
    }
    if (widget_to_toggle.inputEl && widget_to_toggle.inputEl.style) {
      widget_to_toggle.inputEl.style.display = "";
    }
  } else {
    widget_to_toggle.type = HIDDEN_WIDGET_TYPE_SENTINEL_PREFIX + widget_to_toggle.name;
    widget_to_toggle.computeSize = function () {
      return [0, -4];
    };
    // CRITICAL: also set widget.hidden=true. Verified empirically (via Playwright
    // inspection of section_16_weight) that LiteGraph's canvas widget renderer
    // for FLOAT/INT/COMBO widgets respects this flag — the type+computeSize
    // override alone hides STRING multiline (DOM-rendered) but does NOT hide
    // the LAST canvas FLOAT widget in the array.
    widget_to_toggle.hidden = true;
    // Also hide DOM companion if any (defensive — verified that STRING
    // multiline widgets have BOTH .element and .inputEl as <textarea>).
    if (widget_to_toggle.element && widget_to_toggle.element.style) {
      widget_to_toggle.element.style.display = "none";
    }
    if (widget_to_toggle.inputEl && widget_to_toggle.inputEl.style) {
      widget_to_toggle.inputEl.style.display = "none";
    }
  }

  // On show: grow the node to fit the now-visible widgets. On hide: keep
  // current node size (let the user shrink manually if they want).
  const new_height_for_node_after_this_toggle = should_be_visible
    ? Math.max(node.computeSize()[1], node_size_captured_before_this_toggle[1])
    : node.size[1];
  node.setSize([node.size[0], new_height_for_node_after_this_toggle]);
}

function forceFinalNodeHeightRelayoutToFitVisibleWidgets(node) {
  // Equivalent of comfyui-easy-use's updateNodeHeight. Recomputes from
  // current widget visibility state.
  node.setSize([node.size[0], node.computeSize()[1]]);
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
    const this_widget_should_be_visible = this_widget_section_index <= current_section_count_value;
    toggleVisibilityOfOneWidgetOnNodeMatchingComfyuiEasyUsePattern(
      node, widget_descriptor, this_widget_should_be_visible
    );
  }
  forceFinalNodeHeightRelayoutToFitVisibleWidgets(node);
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
      // whenever the user changes it.
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
      // DOM elements may be inserted asynchronously after onNodeCreated.
      // A deferred second pass catches that case.
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
      // section_count widget value. The original widget objects have been
      // replaced, so the cache may have stale entries; allow re-caching
      // by clearing cached entries for THIS node's widgets.
      for (const widget_descriptor of this.widgets || []) {
        if (widget_descriptor && widget_descriptor.name) {
          delete original_widget_props_cache_by_widget_name[widget_descriptor.name];
        }
      }
      updateAllSectionWidgetVisibilityBasedOnCurrentSectionCountValue(this);
      return original_on_configure_return_value;
    };

    // Second wire so the update fires even if some widget implementations
    // bypass the wrapped callback.
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
