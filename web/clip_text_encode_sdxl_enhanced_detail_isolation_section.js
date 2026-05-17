// Frontend extension for CLIPTextEncodeSDXLEnhancedDetailIsolationSection.
//
// Adds a single read-only multiline STRING widget at the bottom of each
// section node called `section_validation_status`. Posts a debounced
// request to the combined section validator endpoint whenever the user
// edits this section's global_text, enhanced_text, or filter toggle.
// The endpoint returns a merged list of:
//   - target-word-not-in-enhanced warnings ("Target word 'foo' not
//     found in enhanced text — will have no isolation effect at that
//     word's position.")
//   - embedding-issue lines ("Embedding NAME not found in system, will
//     be ignored" / "incompatible with SDXL" / "not installed locally,
//     will be stripped (orphan A1111 tag filter)")
// Empty response → status widget shows "(no issues detected for this
// section)".

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const TARGETED_DETAIL_ISOLATION_SECTION_NODE_TYPE_NAME =
  "CLIPTextEncodeSDXLEnhancedDetailIsolationSection";
const SECTION_COMBINED_VALIDATION_STATUS_READ_ONLY_WIDGET_NAME =
  "section_validation_status";
const KEYSTROKE_DEBOUNCE_DELAY_MS_FOR_SECTION_VALIDATOR = 400;
const SECTION_COMBINED_VALIDATOR_HTTP_ENDPOINT_URL =
  "/unified-conditioning-merge/detail_isolation_section_combined_validator";
const PLACEHOLDER_TEXT_BEFORE_FIRST_RUN =
  "(realtime validation — type in global or enhanced text above)";
const NO_ISSUES_FOR_THIS_SECTION_TEXT = "(no issues detected for this section)";

function debounce_section_validator_until_quiet_for_this_many_ms(
  callback_function_to_invoke_after_quiet_period, debounce_delay_milliseconds
) {
  let pending_timer_id = null;
  return function debounced_caller_for_section_validator() {
    const arguments_for_callback = arguments;
    const this_for_callback = this;
    if (pending_timer_id !== null) {
      clearTimeout(pending_timer_id);
    }
    pending_timer_id = setTimeout(function fire_callback_after_quiet_period() {
      pending_timer_id = null;
      callback_function_to_invoke_after_quiet_period.apply(this_for_callback, arguments_for_callback);
    }, debounce_delay_milliseconds);
  };
}

function find_widget_by_name_on_node_or_undefined(node, widget_name_to_find) {
  if (!node.widgets) return undefined;
  for (const widget_descriptor of node.widgets) {
    if (widget_descriptor && widget_descriptor.name === widget_name_to_find) {
      return widget_descriptor;
    }
  }
  return undefined;
}

function gather_current_section_inputs_for_validator_request(node) {
  const global_text_widget = find_widget_by_name_on_node_or_undefined(node, "global_text");
  const enhanced_text_widget = find_widget_by_name_on_node_or_undefined(node, "enhanced_text");
  const filter_toggle_widget = find_widget_by_name_on_node_or_undefined(
    node, "filter_known_a1111_embedding_tags_not_installed_locally"
  );
  return {
    global_text: String(global_text_widget ? (global_text_widget.value || "") : ""),
    enhanced_text: String(enhanced_text_widget ? (enhanced_text_widget.value || "") : ""),
    filter_known_a1111_embedding_tags_not_installed_locally: Boolean(
      filter_toggle_widget ? filter_toggle_widget.value : true
    ),
  };
}

function write_lines_into_section_validation_status_widget_and_redraw(node, message_lines_list) {
  const status_widget_or_undefined = find_widget_by_name_on_node_or_undefined(
    node, SECTION_COMBINED_VALIDATION_STATUS_READ_ONLY_WIDGET_NAME
  );
  if (!status_widget_or_undefined) return;
  const final_displayed_text_value = (message_lines_list && message_lines_list.length > 0)
    ? message_lines_list.join("\n")
    : NO_ISSUES_FOR_THIS_SECTION_TEXT;
  status_widget_or_undefined.value = final_displayed_text_value;
  if (status_widget_or_undefined.inputEl) {
    status_widget_or_undefined.inputEl.value = final_displayed_text_value;
  }
  node.setDirtyCanvas(true, true);
}

async function fetch_combined_section_validator_results_and_update_status_widget(node) {
  const request_body_for_this_section = gather_current_section_inputs_for_validator_request(node);
  try {
    const http_response = await api.fetchApi(SECTION_COMBINED_VALIDATOR_HTTP_ENDPOINT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request_body_for_this_section),
    });
    const parsed_response_body = await http_response.json();
    const message_lines_list = Array.isArray(parsed_response_body && parsed_response_body.messages)
      ? parsed_response_body.messages
      : [];
    write_lines_into_section_validation_status_widget_and_redraw(node, message_lines_list);
  } catch (network_or_parse_error) {
    write_lines_into_section_validation_status_widget_and_redraw(
      node,
      [`[section validator error: ${network_or_parse_error && network_or_parse_error.message}]`]
    );
  }
}

function add_read_only_section_validation_status_widget_to_node_if_not_present(node) {
  const existing_status_widget = find_widget_by_name_on_node_or_undefined(
    node, SECTION_COMBINED_VALIDATION_STATUS_READ_ONLY_WIDGET_NAME
  );
  if (existing_status_widget) return;
  const created_status_widget_descriptor = ComfyWidgets["STRING"](
    node,
    SECTION_COMBINED_VALIDATION_STATUS_READ_ONLY_WIDGET_NAME,
    ["STRING", { multiline: true, default: PLACEHOLDER_TEXT_BEFORE_FIRST_RUN }],
    app,
  );
  const new_widget_object_just_created = created_status_widget_descriptor.widget;
  if (new_widget_object_just_created) {
    new_widget_object_just_created.options = new_widget_object_just_created.options || {};
    new_widget_object_just_created.options.serialize = false;
    if (new_widget_object_just_created.inputEl) {
      new_widget_object_just_created.inputEl.readOnly = true;
      new_widget_object_just_created.inputEl.style.opacity = "0.85";
      new_widget_object_just_created.inputEl.value = PLACEHOLDER_TEXT_BEFORE_FIRST_RUN;
    }
  }
}

function attach_keystroke_listener_to_text_widget_if_not_attached(
  text_widget_object, node, debounced_validator_runner
) {
  if (!text_widget_object) return;
  if (text_widget_object.__section_validator_listener_attached) return;
  text_widget_object.__section_validator_listener_attached = true;
  const previous_widget_callback = text_widget_object.callback;
  text_widget_object.callback = function on_widget_value_changed() {
    if (previous_widget_callback) previous_widget_callback.apply(this, arguments);
    debounced_validator_runner(node);
  };
  if (text_widget_object.inputEl) {
    text_widget_object.inputEl.addEventListener("input", function on_textarea_keystroke() {
      debounced_validator_runner(node);
    });
  }
}

function wrap_boolean_widget_callback_to_fire_validator(
  boolean_widget_object, node, debounced_validator_runner
) {
  if (!boolean_widget_object) return;
  if (boolean_widget_object.__section_validator_callback_wrapped) return;
  boolean_widget_object.__section_validator_callback_wrapped = true;
  const previous_widget_callback = boolean_widget_object.callback;
  boolean_widget_object.callback = function on_boolean_toggle_changed() {
    if (previous_widget_callback) previous_widget_callback.apply(this, arguments);
    debounced_validator_runner(node);
  };
}

app.registerExtension({
  name: "UnifiedConditioningMerge.DetailIsolationSectionCombinedValidator",
  async nodeCreated(node) {
    if (
      !node || !node.constructor
      || node.constructor.type !== TARGETED_DETAIL_ISOLATION_SECTION_NODE_TYPE_NAME
    ) {
      return;
    }
    add_read_only_section_validation_status_widget_to_node_if_not_present(node);
    const debounced_section_validator_runner_for_this_node = (
      debounce_section_validator_until_quiet_for_this_many_ms(
        fetch_combined_section_validator_results_and_update_status_widget,
        KEYSTROKE_DEBOUNCE_DELAY_MS_FOR_SECTION_VALIDATOR,
      )
    );
    const global_text_widget = find_widget_by_name_on_node_or_undefined(node, "global_text");
    const enhanced_text_widget = find_widget_by_name_on_node_or_undefined(node, "enhanced_text");
    const filter_toggle_widget = find_widget_by_name_on_node_or_undefined(
      node, "filter_known_a1111_embedding_tags_not_installed_locally"
    );
    attach_keystroke_listener_to_text_widget_if_not_attached(
      global_text_widget, node, debounced_section_validator_runner_for_this_node
    );
    attach_keystroke_listener_to_text_widget_if_not_attached(
      enhanced_text_widget, node, debounced_section_validator_runner_for_this_node
    );
    wrap_boolean_widget_callback_to_fire_validator(
      filter_toggle_widget, node, debounced_section_validator_runner_for_this_node
    );
    // Initial pass after DOM elements are attached.
    const node_reference_for_initial_pass = node;
    setTimeout(function fire_initial_section_validator_after_dom_attach() {
      fetch_combined_section_validator_results_and_update_status_widget(node_reference_for_initial_pass);
    }, 0);
  },
});
