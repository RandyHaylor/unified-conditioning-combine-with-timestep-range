// Standalone frontend extension for CLIPTextEncodeSDXLV3GlobalAndEnhanced
// that adds a read-only multiline STRING widget at the bottom of the node
// titled `target_word_check_status`. As the user edits any section's
// global_text or enhanced_text, a debounced POST goes to the v3
// target-word-check endpoint (see server_routes.py) and the widget
// updates with per-section "target word 'foo' not found in enhanced
// text" warnings.
//
// Totally independent of the embedding-issue validator widget at the
// other side of the node — different endpoint, different widget,
// different concerns. Embedding validator scans for embedding files /
// SDXL compatibility; this widget verifies global_text words appear
// inside their section's enhanced_text so the cutoff masking can find
// them.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const TARGETED_V3_NODE_TYPE_NAME = "CLIPTextEncodeSDXLV3GlobalAndEnhanced";
const V3_TARGET_WORD_CHECK_STATUS_READ_ONLY_WIDGET_NAME = "target_word_check_status";
const KEYSTROKE_DEBOUNCE_DELAY_MILLISECONDS_FOR_TARGET_WORD_CHECK = 400;
const V3_TARGET_WORD_CHECK_HTTP_ENDPOINT_URL =
  "/unified-conditioning-merge/v3_check_target_words_present_in_enhanced_text";
const PLACEHOLDER_TEXT_BEFORE_FIRST_V3_TARGET_WORD_CHECK_RUN =
  "(realtime target-word check — type in any section's global or enhanced text above)";
const V3_TARGET_WORD_CHECK_NO_ISSUES_TEXT =
  "(no missing target words detected)";

const SECTION_GLOBAL_OR_ENHANCED_TEXT_WIDGET_NAME_REGEX =
  /^section_(\d+)_(global_text|enhanced_text)$/;

function debounce_v3_target_word_check_function_invocation(
  callback_to_invoke_after_quiet_period, debounce_delay_milliseconds
) {
  let pending_timer_id = null;
  return function debounced_caller_for_target_word_check() {
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

function gather_per_section_global_and_enhanced_text_pairs_from_node_widgets(node) {
  // Collects {sectionIndex → {global_text, enhanced_text}} in section_index
  // order (1-based). Sections with no widgets are omitted; missing sub-
  // widgets default to empty string. We then return them as an array
  // ordered by section index so the server can use array index + 1 as
  // the user-visible section number.
  const per_section_text_pairs_keyed_by_section_index = {};
  for (const one_widget_on_node of (node.widgets || [])) {
    if (!one_widget_on_node || !one_widget_on_node.name) continue;
    const widget_name_regex_match = one_widget_on_node.name.match(
      SECTION_GLOBAL_OR_ENHANCED_TEXT_WIDGET_NAME_REGEX
    );
    if (!widget_name_regex_match) continue;
    const section_index_one_based = parseInt(widget_name_regex_match[1], 10);
    const widget_subtype_global_or_enhanced = widget_name_regex_match[2];
    if (!per_section_text_pairs_keyed_by_section_index[section_index_one_based]) {
      per_section_text_pairs_keyed_by_section_index[section_index_one_based] = {
        global_text: "",
        enhanced_text: "",
      };
    }
    per_section_text_pairs_keyed_by_section_index[section_index_one_based][
      widget_subtype_global_or_enhanced
    ] = String(one_widget_on_node.value || "");
  }
  // Also filter by section_count widget so hidden sections aren't reported.
  const section_count_widget = (node.widgets || []).find(
    function find_section_count_widget(w) {
      return w && w.name === "section_count";
    }
  );
  const current_section_count_value = section_count_widget
    ? Math.max(0, Math.floor(Number(section_count_widget.value) || 0))
    : 0;
  const ordered_array_of_section_text_pairs = [];
  for (
    let one_based_section_index = 1;
    one_based_section_index <= current_section_count_value;
    one_based_section_index++
  ) {
    const pair_for_this_section_or_blank =
      per_section_text_pairs_keyed_by_section_index[one_based_section_index]
      || { global_text: "", enhanced_text: "" };
    ordered_array_of_section_text_pairs.push(pair_for_this_section_or_blank);
  }
  return ordered_array_of_section_text_pairs;
}

function write_lines_into_v3_target_word_check_status_widget_and_redraw_node(node, message_lines_list) {
  const status_widget_or_undefined = (node.widgets || []).find(
    function find_status_widget(w) {
      return w && w.name === V3_TARGET_WORD_CHECK_STATUS_READ_ONLY_WIDGET_NAME;
    }
  );
  if (!status_widget_or_undefined) return;
  const final_displayed_text_value = (message_lines_list && message_lines_list.length > 0)
    ? message_lines_list.join("\n")
    : V3_TARGET_WORD_CHECK_NO_ISSUES_TEXT;
  status_widget_or_undefined.value = final_displayed_text_value;
  if (status_widget_or_undefined.inputEl) {
    status_widget_or_undefined.inputEl.value = final_displayed_text_value;
  }
  node.setDirtyCanvas(true, true);
}

async function fetch_v3_target_word_check_results_from_server_and_update_widget(node) {
  const ordered_section_pairs_array = gather_per_section_global_and_enhanced_text_pairs_from_node_widgets(node);
  try {
    const http_response = await api.fetchApi(V3_TARGET_WORD_CHECK_HTTP_ENDPOINT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sections: ordered_section_pairs_array }),
    });
    const parsed_response_body = await http_response.json();
    const message_lines = Array.isArray(parsed_response_body && parsed_response_body.messages)
      ? parsed_response_body.messages
      : [];
    write_lines_into_v3_target_word_check_status_widget_and_redraw_node(node, message_lines);
  } catch (network_or_parse_error) {
    write_lines_into_v3_target_word_check_status_widget_and_redraw_node(
      node,
      [`[v3 target-word check error: ${network_or_parse_error && network_or_parse_error.message}]`]
    );
  }
}

function add_read_only_v3_target_word_check_status_widget_to_node_if_not_already_present(node) {
  const existing_status_widget = (node.widgets || []).find(
    function find_status_widget(w) {
      return w && w.name === V3_TARGET_WORD_CHECK_STATUS_READ_ONLY_WIDGET_NAME;
    }
  );
  if (existing_status_widget) return;
  const created_status_widget_descriptor = ComfyWidgets["STRING"](
    node,
    V3_TARGET_WORD_CHECK_STATUS_READ_ONLY_WIDGET_NAME,
    ["STRING", { multiline: true, default: PLACEHOLDER_TEXT_BEFORE_FIRST_V3_TARGET_WORD_CHECK_RUN }],
    app,
  );
  const new_widget_object_just_created = created_status_widget_descriptor.widget;
  if (new_widget_object_just_created) {
    new_widget_object_just_created.options = new_widget_object_just_created.options || {};
    new_widget_object_just_created.options.serialize = false;
    if (new_widget_object_just_created.inputEl) {
      new_widget_object_just_created.inputEl.readOnly = true;
      new_widget_object_just_created.inputEl.style.opacity = "0.85";
      new_widget_object_just_created.inputEl.value = PLACEHOLDER_TEXT_BEFORE_FIRST_V3_TARGET_WORD_CHECK_RUN;
    }
  }
}

function attach_realtime_input_listener_to_one_v3_text_widget_if_not_already_attached(
  text_widget_object, node, debounced_target_word_check_runner
) {
  if (!text_widget_object) return;
  if (text_widget_object.__v3_target_word_check_listener_attached) return;
  text_widget_object.__v3_target_word_check_listener_attached = true;
  // Wrap the widget's value-setter callback (used by both UI edits and
  // programmatic value assignments) so any change triggers the debounced
  // check.
  const previous_widget_callback = text_widget_object.callback;
  text_widget_object.callback = function on_v3_text_widget_value_changed() {
    if (previous_widget_callback) previous_widget_callback.apply(this, arguments);
    debounced_target_word_check_runner(node);
  };
  // Also attach a DOM `input` listener for keystroke-level updates on the
  // textarea so users don't have to blur to fire the check.
  if (text_widget_object.inputEl) {
    text_widget_object.inputEl.addEventListener("input", function on_textarea_keystroke() {
      debounced_target_word_check_runner(node);
    });
  }
}

app.registerExtension({
  name: "UnifiedConditioningMerge.V3TargetWordInEnhancedCheck",
  async nodeCreated(node) {
    if (!node || !node.constructor || node.constructor.type !== TARGETED_V3_NODE_TYPE_NAME) {
      return;
    }
    add_read_only_v3_target_word_check_status_widget_to_node_if_not_already_present(node);
    const debounced_target_word_check_runner_for_this_node = (
      debounce_v3_target_word_check_function_invocation(
        fetch_v3_target_word_check_results_from_server_and_update_widget,
        KEYSTROKE_DEBOUNCE_DELAY_MILLISECONDS_FOR_TARGET_WORD_CHECK,
      )
    );
    for (const one_widget_on_node of (node.widgets || [])) {
      if (!one_widget_on_node || !one_widget_on_node.name) continue;
      if (SECTION_GLOBAL_OR_ENHANCED_TEXT_WIDGET_NAME_REGEX.test(one_widget_on_node.name)) {
        attach_realtime_input_listener_to_one_v3_text_widget_if_not_already_attached(
          one_widget_on_node, node, debounced_target_word_check_runner_for_this_node
        );
      }
      // Also re-run when section_count changes (sections becoming visible
      // or being hidden change which pairs are checked).
      if (one_widget_on_node.name === "section_count") {
        const previous_section_count_callback = one_widget_on_node.callback;
        one_widget_on_node.callback = function on_section_count_changed_for_target_word_check() {
          if (previous_section_count_callback) {
            previous_section_count_callback.apply(this, arguments);
          }
          debounced_target_word_check_runner_for_this_node(node);
        };
      }
    }
    // Initial pass after a tick so DOM elements are attached.
    const node_reference_for_initial_pass = node;
    setTimeout(function fire_initial_v3_target_word_check_after_dom_attach() {
      fetch_v3_target_word_check_results_from_server_and_update_widget(node_reference_for_initial_pass);
    }, 0);
  },
});
