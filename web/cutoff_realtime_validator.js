// Real-time embedding validator for the
// CLIPTextEncodeWithCutoffRegionSeparation node.
//
// Adds a read-only multiline STRING widget at the bottom of the node
// titled `validation_status`. As the user types in any
// `section_N_text` widget, a debounced POST goes to the plugin's server
// route and the widget is updated with the resulting per-embedding
// classification lines:
//     embedding:NAME not found on system
//     embedding:NAME incompatible with SDXL
//
// Validation is SDXL-targeted: the plugin's whole intent is SDXL CLIP
// stream handling, so the server marks any embedding file lacking BOTH
// a 768-dim and a 1280-dim tensor as "incompatible with SDXL".
//
// Performance notes:
//   - 400ms debounce per node, so a burst of typing produces at most
//     one HTTP roundtrip per pause.
//   - The validator widget is marked `options.serialize = false`,
//     so its computed value does not pollute the saved workflow JSON.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const NODE_TYPE_NAMES_THIS_EXTENSION_TARGETS = new Set([
  "CLIPTextEncodeWithCutoffRegionSeparation",
  "CLIPTextEncodeSDXLV2WithIsolationAmount",
]);
// Legacy single-value alias preserved for any external code that imported
// it. New checks use NODE_TYPE_NAMES_THIS_EXTENSION_TARGETS.has(name).
const NODE_TYPE_NAME_THIS_EXTENSION_TARGETS = "CLIPTextEncodeWithCutoffRegionSeparation";
const VALIDATION_STATUS_READ_ONLY_WIDGET_NAME = "validation_status";
const KEYSTROKE_DEBOUNCE_DELAY_MILLISECONDS = 400;
const PROMPT_VALIDATION_HTTP_ENDPOINT_URL = "/unified-conditioning-merge/validate_prompt_embeddings_sdxl";
const PLACEHOLDER_TEXT_BEFORE_FIRST_VALIDATION_RUN = "(realtime validation — type in any section above)";
const NO_ISSUES_TEXT = "(no embedding issues detected)";

function debounce_function_invocation_until_no_call_for_this_many_ms(
  callback_to_invoke_after_quiet_period, debounce_delay_milliseconds
) {
  let pending_timer_id = null;
  return function debounced_caller() {
    const arguments_for_callback = arguments;
    const this_for_callback = this;
    if (pending_timer_id !== null) {
      clearTimeout(pending_timer_id);
    }
    pending_timer_id = setTimeout(function fire_callback_after_quiet_period() {
      pending_timer_id = null;
      callback_to_invoke_after_quiet_period.apply(this_for_callback, arguments_for_callback);
    }, debounce_delay_milliseconds);
  };
}

function widget_is_one_of_the_per_section_prompt_text_widgets(widget_object) {
  // Matches both v1 `section_N_text` and v2 `region_N_text` widget names.
  if (!widget_object || !widget_object.name) return false;
  const ends_with_text_suffix = widget_object.name.endsWith("_text");
  if (!ends_with_text_suffix) return false;
  return (
    widget_object.name.startsWith("section_")
    || widget_object.name.startsWith("region_")
  );
}

function gather_current_text_values_from_all_section_text_widgets_on_node(node) {
  const collected_text_values_in_widget_array_order = [];
  for (const one_widget_on_node of (node.widgets || [])) {
    if (widget_is_one_of_the_per_section_prompt_text_widgets(one_widget_on_node)) {
      collected_text_values_in_widget_array_order.push(String(one_widget_on_node.value || ""));
    }
  }
  return collected_text_values_in_widget_array_order;
}

function read_one_widget_value_by_name_or_default(node, widget_name_to_find, fallback_value) {
  const matching_widget_or_undefined = (node.widgets || []).find(
    function find_named_widget(w) {
      return w && w.name === widget_name_to_find;
    }
  );
  if (!matching_widget_or_undefined) return fallback_value;
  return matching_widget_or_undefined.value;
}

function write_lines_into_validation_status_widget_and_redraw_node(node, message_lines_list) {
  const validation_widget_or_undefined = (node.widgets || []).find(
    function find_status_widget(w) {
      return w && w.name === VALIDATION_STATUS_READ_ONLY_WIDGET_NAME;
    }
  );
  if (!validation_widget_or_undefined) return;
  const text_to_display_in_widget = (message_lines_list.length === 0)
    ? NO_ISSUES_TEXT
    : message_lines_list.join("\n");
  validation_widget_or_undefined.value = text_to_display_in_widget;
  if (validation_widget_or_undefined.inputEl) {
    validation_widget_or_undefined.inputEl.value = text_to_display_in_widget;
  }
  node.setDirtyCanvas(true, true);
}

async function fetch_validation_results_from_server_and_update_widget_for_node(node) {
  const all_section_text_values = gather_current_text_values_from_all_section_text_widgets_on_node(node);
  const filter_known_a1111_widget_value = Boolean(
    read_one_widget_value_by_name_or_default(node, "filter_known_a1111_embedding_tags_not_installed_locally", true)
  );
  try {
    const http_response = await api.fetchApi(PROMPT_VALIDATION_HTTP_ENDPOINT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt_texts: all_section_text_values,
        filter_known_a1111_embedding_tags_not_installed_locally: filter_known_a1111_widget_value,
      }),
    });
    const parsed_response_body = await http_response.json();
    const message_lines = Array.isArray(parsed_response_body && parsed_response_body.messages)
      ? parsed_response_body.messages
      : [];
    write_lines_into_validation_status_widget_and_redraw_node(node, message_lines);
  } catch (validation_request_error) {
    write_lines_into_validation_status_widget_and_redraw_node(
      node,
      [`[validation request failed: ${validation_request_error.message || validation_request_error}]`]
    );
  }
}

function attach_realtime_input_listener_to_one_text_widget_if_not_already_attached(
  text_widget_object, node, debounced_validation_runner
) {
  if (!text_widget_object || !text_widget_object.inputEl) return;
  if (text_widget_object.__unified_conditioning_merge_realtime_listener_already_attached) return;
  text_widget_object.inputEl.addEventListener("input", function on_keystroke_in_text_widget() {
    debounced_validation_runner(node);
  });
  text_widget_object.__unified_conditioning_merge_realtime_listener_already_attached = true;
}

function add_read_only_validation_status_widget_to_node_if_not_already_present(node) {
  const already_has_validation_status_widget = (node.widgets || []).some(
    function check_for_existing_widget(w) {
      return w && w.name === VALIDATION_STATUS_READ_ONLY_WIDGET_NAME;
    }
  );
  if (already_has_validation_status_widget) return;
  if (!ComfyWidgets || !ComfyWidgets["STRING"]) return;

  const created_widget_wrapper = ComfyWidgets["STRING"](
    node,
    VALIDATION_STATUS_READ_ONLY_WIDGET_NAME,
    [
      "STRING",
      {
        multiline: true,
        default: PLACEHOLDER_TEXT_BEFORE_FIRST_VALIDATION_RUN,
      },
    ],
    app
  );
  const validation_status_widget = created_widget_wrapper && created_widget_wrapper.widget;
  if (!validation_status_widget) return;

  if (validation_status_widget.inputEl) {
    validation_status_widget.inputEl.readOnly = true;
    validation_status_widget.inputEl.style.opacity = "0.75";
    validation_status_widget.inputEl.style.fontFamily =
      "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace";
  }

  validation_status_widget.options = validation_status_widget.options || {};
  validation_status_widget.options.serialize = false;
}

app.registerExtension({
  name: "UnifiedConditioningMerge.CutoffRealtimeValidator",
  async nodeCreated(node) {
    if (!node || !node.constructor || !NODE_TYPE_NAMES_THIS_EXTENSION_TARGETS.has(node.constructor.type)) {
      return;
    }

    // Zoom-effect header is injected by the OTHER frontend extension
    // (web/clip_text_encode_with_cutoff_region_separation.js) which owns
    // section headers and the zoom group header. We don't duplicate it here.

    add_read_only_validation_status_widget_to_node_if_not_already_present(node);

    const debounced_validation_runner_for_this_node = debounce_function_invocation_until_no_call_for_this_many_ms(
      fetch_validation_results_from_server_and_update_widget_for_node,
      KEYSTROKE_DEBOUNCE_DELAY_MILLISECONDS
    );

    for (const one_widget_on_node of (node.widgets || [])) {
      if (widget_is_one_of_the_per_section_prompt_text_widgets(one_widget_on_node)) {
        attach_realtime_input_listener_to_one_text_widget_if_not_already_attached(
          one_widget_on_node, node, debounced_validation_runner_for_this_node
        );
      }
      // Also fire validation when the user toggles the filter setting so the
      // displayed warnings stay in sync with the runtime behavior.
      if (
        one_widget_on_node
        && one_widget_on_node.name === "filter_known_a1111_embedding_tags_not_installed_locally"
      ) {
        const original_widget_callback_function = one_widget_on_node.callback;
        one_widget_on_node.callback = function on_filter_toggle_change() {
          if (original_widget_callback_function) {
            original_widget_callback_function.apply(this, arguments);
          }
          debounced_validation_runner_for_this_node(node);
        };
      }
    }

    // Kick off an initial validation pass so the user sees results
    // immediately (without having to type anything first).
    fetch_validation_results_from_server_and_update_widget_for_node(node);
  },
});
