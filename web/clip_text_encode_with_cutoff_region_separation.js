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

const NODE_TYPE_NAME_FOR_THIS_EXTENSION = "CLIPTextEncodeWithCutoffRegionSeparation";

const SECTION_WIDGET_NAME_REGEX = /^section_(\d+)_(text|isolate|weight|clip)$/;
const SECTION_TEXT_WIDGET_NAME_REGEX = /^section_(\d+)_text$/;
const SECTION_HEADER_WIDGET_NAME_PREFIX = "__section_header_for_index_";
const SECTION_HEADER_WIDGET_NAME_REGEX = /^__section_header_for_index_(\d+)$/;

const HIDDEN_WIDGET_TYPE_SENTINEL_PREFIX = "cutoffSectionsHidden:";

// One global cache keyed by widget.name. The widget objects themselves get
// re-created on workflow load, so storing the originals on the widget would
// not survive a configure pass.
const original_widget_props_cache_by_widget_name = {};

function ensureSectionHeaderWidgetsAreInsertedBeforeEachSectionTextWidget(node) {
  if (!node.widgets) return;
  // Walk by index because we mutate node.widgets[]. Skip past header+text
  // after inserting so we don't re-process them.
  let widget_index_iterator_position = 0;
  while (widget_index_iterator_position < node.widgets.length) {
    const candidate_widget_at_current_position = node.widgets[widget_index_iterator_position];
    const text_widget_name_regex_match = candidate_widget_at_current_position && candidate_widget_at_current_position.name
      ? candidate_widget_at_current_position.name.match(SECTION_TEXT_WIDGET_NAME_REGEX)
      : null;
    if (!text_widget_name_regex_match) {
      widget_index_iterator_position++;
      continue;
    }
    const section_index_for_this_text_widget = parseInt(text_widget_name_regex_match[1], 10);
    const expected_header_widget_name_for_this_section_index = SECTION_HEADER_WIDGET_NAME_PREFIX + section_index_for_this_text_widget;
    const previous_widget_or_null = widget_index_iterator_position > 0
      ? node.widgets[widget_index_iterator_position - 1]
      : null;
    if (previous_widget_or_null && previous_widget_or_null.name === expected_header_widget_name_for_this_section_index) {
      // Header already in place.
      widget_index_iterator_position++;
      continue;
    }
    const new_section_header_widget_to_insert = {
      name: expected_header_widget_name_for_this_section_index,
      type: "custom",
      value: "",
      __section_index_for_header_display_only: section_index_for_this_text_widget,
      options: { serialize: false },
      draw(canvas_context, owning_node, widget_width_pixels, y_top_pixels, widget_height_pixels) {
        canvas_context.save();
        canvas_context.fillStyle = "#9aa";
        canvas_context.font = "bold 11px Arial, sans-serif";
        canvas_context.textBaseline = "middle";
        const header_label_text = "── section " + this.__section_index_for_header_display_only + " ──";
        canvas_context.fillText(header_label_text, 12, y_top_pixels + widget_height_pixels / 2);
        canvas_context.restore();
      },
      computeSize(available_widget_width_pixels) {
        return [available_widget_width_pixels, 16];
      },
    };
    node.widgets.splice(widget_index_iterator_position, 0, new_section_header_widget_to_insert);
    // Skip past the newly-inserted header AND the text widget that follows.
    widget_index_iterator_position += 2;
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
    // Always hide the mask_token widget. The value stays at its default
    // (empty string) and the Python side reads it from kwargs as usual.
    // Cutoff treats empty mask_token as the default end-of-text token,
    // which is what we want for phrase-level decontamination, so there's
    // no reason to expose it in the UI.
    if (widget_descriptor.name === "mask_token") {
      toggleVisibilityOfOneWidgetOnNodeMatchingComfyuiEasyUsePattern(
        node, widget_descriptor, false
      );
      continue;
    }
    const regex_match_for_section_widget_name = widget_descriptor.name.match(SECTION_WIDGET_NAME_REGEX);
    const regex_match_for_section_header_widget_name = widget_descriptor.name.match(SECTION_HEADER_WIDGET_NAME_REGEX);
    let this_widget_section_index_or_null = null;
    if (regex_match_for_section_widget_name) {
      this_widget_section_index_or_null = parseInt(regex_match_for_section_widget_name[1], 10);
    } else if (regex_match_for_section_header_widget_name) {
      this_widget_section_index_or_null = parseInt(regex_match_for_section_header_widget_name[1], 10);
    } else {
      continue;
    }
    const this_widget_should_be_visible = this_widget_section_index_or_null <= current_section_count_value;
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

      ensureSectionHeaderWidgetsAreInsertedBeforeEachSectionTextWidget(this);
      updateAllSectionWidgetVisibilityBasedOnCurrentSectionCountValue(this);
      // DOM elements may be inserted asynchronously after onNodeCreated.
      // A deferred second pass catches that case.
      const node_reference_for_deferred_update = this;
      setTimeout(function () {
        ensureSectionHeaderWidgetsAreInsertedBeforeEachSectionTextWidget(node_reference_for_deferred_update);
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
      ensureSectionHeaderWidgetsAreInsertedBeforeEachSectionTextWidget(this);
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
