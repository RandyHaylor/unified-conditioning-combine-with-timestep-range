// Frontend extension for CLIPTextEncodeSDXLV2WithIsolationAmount.
//
// Hides region_N_* widget groups whose index N is greater than the current
// `region_count` widget value, and ZEROES out any widget that becomes hidden
// (text -> "", numeric -> 0). Pattern mirrors v1's
// clip_text_encode_with_cutoff_region_separation.js but adapted to v2's
// region widget naming.
//
// Per-region widget names (six per region):
//   region_N_text
//   region_N_weight
//   region_N_isolation_amount
//   region_N_clip_l_strength
//   region_N_clip_g_strength
//   region_N_weight_from_other_isolated_regions

import { app } from "../../scripts/app.js";

const NODE_TYPE_NAME_FOR_THIS_EXTENSION = "CLIPTextEncodeSDXLV2WithIsolationAmount";

const REGION_WIDGET_NAME_REGEX = /^region_(\d+)_(text|weight|isolation_amount|clip_l_strength|clip_g_strength|weight_from_other_isolated_regions)$/;
const REGION_TEXT_WIDGET_NAME_REGEX = /^region_(\d+)_text$/;
const REGION_HEADER_WIDGET_NAME_PREFIX = "__v2_region_header_for_index_";
const REGION_HEADER_WIDGET_NAME_REGEX = /^__v2_region_header_for_index_(\d+)$/;

const HIDDEN_WIDGET_TYPE_SENTINEL_PREFIX = "v2RegionHidden:";

const ZOOM_GROUP_HEADER_WIDGET_NAME = "__v2_zoom_group_header_static";
const ZOOM_GROUP_HEADER_DISPLAY_LABEL_TEXT = "── zoom effect: based on SDXL CLIP source, target image settings ──";
const ZOOM_WIDGET_NAME_TO_INSERT_HEADER_BEFORE = "zoom";

function ensureStaticZoomGroupHeaderWidgetIsInsertedBeforeTheZoomWidget(node) {
  if (!node.widgets) return;
  const existing_header_index = node.widgets.findIndex(
    (w) => w && w.name === ZOOM_GROUP_HEADER_WIDGET_NAME,
  );
  if (existing_header_index >= 0) return;
  const zoom_widget_index = node.widgets.findIndex(
    (w) => w && w.name === ZOOM_WIDGET_NAME_TO_INSERT_HEADER_BEFORE,
  );
  if (zoom_widget_index < 0) return;
  const zoom_group_header_widget_to_insert = {
    name: ZOOM_GROUP_HEADER_WIDGET_NAME,
    type: "custom",
    value: "",
    options: { serialize: false },
    draw(canvas_context, owning_node, widget_width_pixels, y_top_pixels, widget_height_pixels) {
      canvas_context.save();
      canvas_context.fillStyle = "#9aa";
      canvas_context.font = "bold 11px Arial, sans-serif";
      canvas_context.textBaseline = "middle";
      canvas_context.fillText(
        ZOOM_GROUP_HEADER_DISPLAY_LABEL_TEXT,
        12,
        y_top_pixels + widget_height_pixels / 2,
      );
      canvas_context.restore();
    },
    computeSize(available_widget_width_pixels) {
      return [available_widget_width_pixels, 16];
    },
  };
  node.widgets.splice(zoom_widget_index, 0, zoom_group_header_widget_to_insert);
}

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

function ensureRegionHeaderWidgetsAreInsertedBeforeEachRegionTextWidget(node) {
  if (!node.widgets) return;
  let widget_index_iterator_position = 0;
  while (widget_index_iterator_position < node.widgets.length) {
    const candidate_widget = node.widgets[widget_index_iterator_position];
    const text_widget_name_regex_match = candidate_widget && candidate_widget.name
      ? candidate_widget.name.match(REGION_TEXT_WIDGET_NAME_REGEX)
      : null;
    if (!text_widget_name_regex_match) {
      widget_index_iterator_position++;
      continue;
    }
    const region_index_for_this_text_widget = parseInt(text_widget_name_regex_match[1], 10);
    const expected_header_widget_name = REGION_HEADER_WIDGET_NAME_PREFIX + region_index_for_this_text_widget;
    const previous_widget_or_null = widget_index_iterator_position > 0
      ? node.widgets[widget_index_iterator_position - 1]
      : null;
    if (previous_widget_or_null && previous_widget_or_null.name === expected_header_widget_name) {
      widget_index_iterator_position++;
      continue;
    }
    const new_region_header_widget_to_insert = {
      name: expected_header_widget_name,
      type: "custom",
      value: "",
      __region_index_for_header_display_only: region_index_for_this_text_widget,
      options: { serialize: false },
      draw(canvas_context, owning_node, widget_width_pixels, y_top_pixels, widget_height_pixels) {
        canvas_context.save();
        canvas_context.fillStyle = "#9aa";
        canvas_context.font = "bold 11px Arial, sans-serif";
        canvas_context.textBaseline = "bottom";
        const header_label_text = "── region " + this.__region_index_for_header_display_only + " ──";
        const text_baseline_y_position = y_top_pixels + widget_height_pixels - 2;
        canvas_context.fillText(header_label_text, 12, text_baseline_y_position);
        canvas_context.restore();
      },
      computeSize(available_widget_width_pixels) {
        return [available_widget_width_pixels, 24];
      },
    };
    node.widgets.splice(widget_index_iterator_position, 0, new_region_header_widget_to_insert);
    widget_index_iterator_position += 2;
  }
}

function toggleVisibilityOfOneWidgetOnNodeMatchingComfyuiEasyUsePattern(node, widget_to_toggle, should_be_visible) {
  if (!widget_to_toggle) return;

  if (!original_widget_props_cache_by_widget_name[widget_to_toggle.name]) {
    original_widget_props_cache_by_widget_name[widget_to_toggle.name] = {
      original_type: widget_to_toggle.type,
      original_compute_size_function: widget_to_toggle.computeSize,
    };
  }
  const cached_original_props = original_widget_props_cache_by_widget_name[widget_to_toggle.name];

  const node_size_before_toggle = [node.size[0], node.size[1]];

  if (should_be_visible) {
    widget_to_toggle.type = cached_original_props.original_type;
    delete widget_to_toggle.computeSize;
    widget_to_toggle.hidden = false;
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
    widget_to_toggle.hidden = true;
    if (widget_to_toggle.element && widget_to_toggle.element.style) {
      widget_to_toggle.element.style.display = "none";
    }
    if (widget_to_toggle.inputEl && widget_to_toggle.inputEl.style) {
      widget_to_toggle.inputEl.style.display = "none";
    }
  }

  const new_height_for_node = should_be_visible
    ? Math.max(node.computeSize()[1], node_size_before_toggle[1])
    : node.size[1];
  node.setSize([node.size[0], new_height_for_node]);
}

function zeroOutOneRegionWidgetValueIfBecomingHidden(widget_to_zero_out) {
  if (!widget_to_zero_out || !widget_to_zero_out.name) return;
  const widget_name_regex_match = widget_to_zero_out.name.match(REGION_WIDGET_NAME_REGEX);
  if (!widget_name_regex_match) return;
  const widget_subtype_segment = widget_name_regex_match[2];
  if (widget_subtype_segment === "text") {
    widget_to_zero_out.value = "";
    // Sync DOM textarea if it exists (STRING multiline widgets render to a textarea).
    if (widget_to_zero_out.inputEl) {
      widget_to_zero_out.inputEl.value = "";
    }
    if (widget_to_zero_out.element && widget_to_zero_out.element.tagName === "TEXTAREA") {
      widget_to_zero_out.element.value = "";
    }
  } else {
    // All other v2 region widgets are FLOAT — zero them.
    widget_to_zero_out.value = 0;
  }
}

function forceFinalNodeHeightRelayoutToFitVisibleWidgets(node) {
  node.setSize([node.size[0], node.computeSize()[1]]);
}

function updateAllRegionWidgetVisibilityBasedOnCurrentRegionCountValue(node) {
  const region_count_widget = findWidgetByNameOnNodeOrUndefined(node, "region_count");
  if (!region_count_widget) return;
  const current_region_count_value = Math.max(
    0, Math.floor(Number(region_count_widget.value) || 0)
  );
  for (const widget_descriptor of node.widgets || []) {
    if (!widget_descriptor || !widget_descriptor.name) continue;
    const regex_match_for_region_widget = widget_descriptor.name.match(REGION_WIDGET_NAME_REGEX);
    const regex_match_for_region_header = widget_descriptor.name.match(REGION_HEADER_WIDGET_NAME_REGEX);
    let this_widget_region_index_or_null = null;
    if (regex_match_for_region_widget) {
      this_widget_region_index_or_null = parseInt(regex_match_for_region_widget[1], 10);
    } else if (regex_match_for_region_header) {
      this_widget_region_index_or_null = parseInt(regex_match_for_region_header[1], 10);
    } else {
      continue;
    }
    const this_widget_should_be_visible = this_widget_region_index_or_null <= current_region_count_value;
    const widget_was_visible_before_toggle = !widget_descriptor.hidden
      && (
        !widget_descriptor.type
        || !String(widget_descriptor.type).startsWith(HIDDEN_WIDGET_TYPE_SENTINEL_PREFIX)
      );
    toggleVisibilityOfOneWidgetOnNodeMatchingComfyuiEasyUsePattern(
      node, widget_descriptor, this_widget_should_be_visible
    );
    // Zero out values only on the visible->hidden transition. Don't touch
    // values when a widget is already hidden, and don't touch values when
    // a widget becomes visible (user may want to re-edit).
    if (widget_was_visible_before_toggle && !this_widget_should_be_visible) {
      zeroOutOneRegionWidgetValueIfBecomingHidden(widget_descriptor);
    }
  }
  forceFinalNodeHeightRelayoutToFitVisibleWidgets(node);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "CLIPTextEncodeSDXLV2WithIsolationAmount.RegionVisibility",
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== NODE_TYPE_NAME_FOR_THIS_EXTENSION) return;

    const original_on_node_created_function = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      this.serialize_widgets = true;
      const original_on_node_created_return_value = original_on_node_created_function
        ? original_on_node_created_function.apply(this, arguments)
        : undefined;

      const region_count_widget = findWidgetByNameOnNodeOrUndefined(this, "region_count");
      if (region_count_widget) {
        const previous_region_count_widget_callback = region_count_widget.callback;
        const node_reference_for_callback = this;
        region_count_widget.callback = function () {
          const previous_callback_return_value = previous_region_count_widget_callback
            ? previous_region_count_widget_callback.apply(this, arguments)
            : undefined;
          updateAllRegionWidgetVisibilityBasedOnCurrentRegionCountValue(node_reference_for_callback);
          return previous_callback_return_value;
        };
      }

      ensureStaticZoomGroupHeaderWidgetIsInsertedBeforeTheZoomWidget(this);
      ensureRegionHeaderWidgetsAreInsertedBeforeEachRegionTextWidget(this);
      updateAllRegionWidgetVisibilityBasedOnCurrentRegionCountValue(this);
      const node_reference_for_deferred_update = this;
      setTimeout(function () {
        ensureStaticZoomGroupHeaderWidgetIsInsertedBeforeTheZoomWidget(node_reference_for_deferred_update);
        ensureRegionHeaderWidgetsAreInsertedBeforeEachRegionTextWidget(node_reference_for_deferred_update);
        updateAllRegionWidgetVisibilityBasedOnCurrentRegionCountValue(node_reference_for_deferred_update);
      }, 0);
      return original_on_node_created_return_value;
    };

    const original_on_configure_function = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (_serialized_node_data) {
      const original_on_configure_return_value = original_on_configure_function
        ? original_on_configure_function.apply(this, arguments)
        : undefined;
      for (const widget_descriptor of this.widgets || []) {
        if (widget_descriptor && widget_descriptor.name) {
          delete original_widget_props_cache_by_widget_name[widget_descriptor.name];
        }
      }
      ensureStaticZoomGroupHeaderWidgetIsInsertedBeforeTheZoomWidget(this);
      ensureRegionHeaderWidgetsAreInsertedBeforeEachRegionTextWidget(this);
      updateAllRegionWidgetVisibilityBasedOnCurrentRegionCountValue(this);
      return original_on_configure_return_value;
    };

    const original_on_widget_changed_function = nodeType.prototype.onWidgetChanged;
    nodeType.prototype.onWidgetChanged = function (changed_widget_name, _new_widget_value, _old_widget_value, _changed_widget) {
      const original_on_widget_changed_return_value = original_on_widget_changed_function
        ? original_on_widget_changed_function.apply(this, arguments)
        : undefined;
      if (changed_widget_name === "region_count") {
        updateAllRegionWidgetVisibilityBasedOnCurrentRegionCountValue(this);
      }
      return original_on_widget_changed_return_value;
    };
  },
});
