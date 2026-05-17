// Frontend extension for CLIPTextEncodeSDXLEnhancedInlineTagged.
//
// Inserts two canvas-drawn header label widgets above the
// `inline_tagged_prompt_text` widget to display syntax instructions
// directly on the node face (not as placeholder text that disappears
// when the user starts typing). Also inserts the standard zoom-effect
// group header above the `zoom` widget.

import { app } from "../../scripts/app.js";

const TARGETED_INLINE_TAGGED_NODE_TYPE_NAME = "CLIPTextEncodeSDXLEnhancedInlineTagged";

const INLINE_TAGGED_PROMPT_TEXT_WIDGET_NAME = "inline_tagged_prompt_text";
const INLINE_TAGGED_PROMPT_HEADER_WIDGET_NAME = "__inline_tagged_prompt_header_static";
const INLINE_TAGGED_PROMPT_HEADER_LABEL_LINES_LIST_ABOVE_FIELD = [
  "── inline tag syntax (tags are stripped before encoding) ──",
  "<REGION>...</REGION>  marks a cutoff region (its body becomes part of the natural prompt).",
  "<DETAIL>words</DETAIL>  inside a REGION marks words masked from OTHER regions' attention.",
  "Text outside REGION blocks is passthrough (no isolation).",
];

const ZOOM_GROUP_HEADER_WIDGET_NAME = "__inline_tagged_zoom_group_header_static";
const ZOOM_GROUP_HEADER_DISPLAY_LABEL_TEXT =
  "── zoom effect: based on SDXL CLIP source, target image settings ──";
const ZOOM_WIDGET_NAME_TO_INSERT_HEADER_BEFORE = "zoom";

const HEADER_VERTICAL_PIXEL_PADDING_PER_LINE = 14;
const HEADER_BOTTOM_PIXEL_PADDING_BELOW_LAST_LINE = 6;

function find_widget_by_name_on_node_or_undefined(node, widget_name_to_find) {
  if (!node.widgets) return undefined;
  for (const widget_descriptor of node.widgets) {
    if (widget_descriptor && widget_descriptor.name === widget_name_to_find) {
      return widget_descriptor;
    }
  }
  return undefined;
}

function ensure_inline_tagged_prompt_header_widget_is_inserted_directly_above_prompt_field(node) {
  if (!node.widgets) return;
  if (find_widget_by_name_on_node_or_undefined(node, INLINE_TAGGED_PROMPT_HEADER_WIDGET_NAME)) return;
  const prompt_widget_array_index = node.widgets.findIndex(
    function find_prompt_widget(w) {
      return w && w.name === INLINE_TAGGED_PROMPT_TEXT_WIDGET_NAME;
    }
  );
  if (prompt_widget_array_index < 0) return;
  const computed_total_header_height_in_pixels = (
    INLINE_TAGGED_PROMPT_HEADER_LABEL_LINES_LIST_ABOVE_FIELD.length * HEADER_VERTICAL_PIXEL_PADDING_PER_LINE
    + HEADER_BOTTOM_PIXEL_PADDING_BELOW_LAST_LINE
  );
  const inline_tagged_prompt_header_widget_to_insert = {
    name: INLINE_TAGGED_PROMPT_HEADER_WIDGET_NAME,
    type: "custom",
    value: "",
    options: { serialize: false },
    draw(canvas_context, owning_node, widget_width_pixels, y_top_pixels, widget_height_pixels) {
      canvas_context.save();
      canvas_context.font = "11px Arial, sans-serif";
      canvas_context.textBaseline = "top";
      for (
        let line_index_into_label_lines_list = 0;
        line_index_into_label_lines_list < INLINE_TAGGED_PROMPT_HEADER_LABEL_LINES_LIST_ABOVE_FIELD.length;
        line_index_into_label_lines_list++
      ) {
        const one_line_label_text = INLINE_TAGGED_PROMPT_HEADER_LABEL_LINES_LIST_ABOVE_FIELD[line_index_into_label_lines_list];
        // Make the first line (the section divider) brighter; subsequent
        // lines (the actual instructions) slightly dimmer so the header
        // doesn't visually compete with the text the user is typing.
        canvas_context.fillStyle = line_index_into_label_lines_list === 0 ? "#9aa" : "#bbc";
        if (line_index_into_label_lines_list === 0) {
          canvas_context.font = "bold 11px Arial, sans-serif";
        } else {
          canvas_context.font = "11px Arial, sans-serif";
        }
        canvas_context.fillText(
          one_line_label_text,
          12,
          y_top_pixels + 2 + line_index_into_label_lines_list * HEADER_VERTICAL_PIXEL_PADDING_PER_LINE,
        );
      }
      canvas_context.restore();
    },
    computeSize(available_widget_width_pixels) {
      return [available_widget_width_pixels, computed_total_header_height_in_pixels];
    },
  };
  node.widgets.splice(prompt_widget_array_index, 0, inline_tagged_prompt_header_widget_to_insert);
}

function ensure_zoom_group_header_widget_is_inserted_above_zoom_widget(node) {
  if (!node.widgets) return;
  if (find_widget_by_name_on_node_or_undefined(node, ZOOM_GROUP_HEADER_WIDGET_NAME)) return;
  const zoom_widget_array_index = node.widgets.findIndex(
    function find_zoom_widget(w) {
      return w && w.name === ZOOM_WIDGET_NAME_TO_INSERT_HEADER_BEFORE;
    }
  );
  if (zoom_widget_array_index < 0) return;
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
  node.widgets.splice(zoom_widget_array_index, 0, zoom_group_header_widget_to_insert);
}

function force_node_height_recompute_to_fit_inserted_headers(node) {
  node.setSize([node.size[0], node.computeSize()[1]]);
  node.setDirtyCanvas(true, true);
}

app.registerExtension({
  name: "UnifiedConditioningMerge.InlineTaggedHeaders",
  async nodeCreated(node) {
    if (!node || !node.constructor || node.constructor.type !== TARGETED_INLINE_TAGGED_NODE_TYPE_NAME) {
      return;
    }
    ensure_inline_tagged_prompt_header_widget_is_inserted_directly_above_prompt_field(node);
    ensure_zoom_group_header_widget_is_inserted_above_zoom_widget(node);
    force_node_height_recompute_to_fit_inserted_headers(node);
    // Deferred second pass for DOM-attached widgets that arrive after
    // nodeCreated fires.
    const node_reference_for_deferred = node;
    setTimeout(function fire_deferred_header_insertion() {
      ensure_inline_tagged_prompt_header_widget_is_inserted_directly_above_prompt_field(node_reference_for_deferred);
      ensure_zoom_group_header_widget_is_inserted_above_zoom_widget(node_reference_for_deferred);
      force_node_height_recompute_to_fit_inserted_headers(node_reference_for_deferred);
    }, 0);
  },
});
