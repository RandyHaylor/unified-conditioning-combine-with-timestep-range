"""
HTTP routes exposed by this plugin for its frontend extensions.

Routes registered:

  POST /unified-conditioning-merge/validate_prompt_embeddings_sdxl
    Request body JSON:
        {"prompt_texts": ["section1 text", "section2 text", ...]}
    Response body JSON:
        {"messages": [
            "embedding:NAME not found on system",
            "embedding:NAME incompatible with SDXL",
            ...
         ]}
    For each `embedding:NAME` reference found in any of the prompt texts
    (whether the explicit `embedding:NAME` form or an A1111-style bare
    tag that matches a file in the embeddings directory), one classification
    line is emitted:
      - "not found on system" — no file with that stem exists.
      - "incompatible with SDXL" — file exists but its tensor dims don't
        include BOTH 768 (CLIP-L) AND 1280 (CLIP-G).
      - (nothing) — file exists and contains both required dims.

  POST /unified-conditioning-merge/rescan_embeddings_directory
    Forces the embedding-folder scan cache to invalidate and rebuild.
    Response body JSON: {"indexed_file_count": <int>}
    Use after adding/removing embedding files while ComfyUI is running.
"""

import logging
import re

from aiohttp import web
from server import PromptServer

from .embedding_index_scanner import (
    get_cached_or_build_embedding_lowercase_stem_to_index_entry_map,
    invalidate_cached_embedding_filename_to_dim_index,
    is_embedding_file_fully_compatible_with_sdxl_based_on_its_tensor_last_dim_set,
)


EXPLICIT_EMBEDDING_REFERENCE_REGEX_PATTERN = re.compile(r"embedding:([\w./\\-]+)")


def _collect_all_embedding_references_appearing_in_one_prompt_text(
    one_prompt_text, embedding_lowercase_stem_to_index_entry_map
):
    """
    Returns a set of (name_used_in_prompt, lowercase_stem_for_index_lookup,
    bool_is_a1111_bare_style) tuples. Picks up:
      1. Explicit `embedding:NAME` references regardless of whether the file
         actually exists (so we can flag missing ones).
      2. A1111-style bare comma-separated tags (or parenthesized
         `(tag:weight)` forms) but ONLY when they match a known indexed
         filename stem — otherwise plain tags would all light up as
         "embedding not found".
    """
    references_set = set()
    if not one_prompt_text:
        return references_set

    for explicit_match_object in EXPLICIT_EMBEDDING_REFERENCE_REGEX_PATTERN.finditer(one_prompt_text):
        name_string_used_in_prompt = explicit_match_object.group(1)
        references_set.add(
            (name_string_used_in_prompt, name_string_used_in_prompt.lower(), False)
        )

    for raw_comma_separated_part in one_prompt_text.split(","):
        stripped_part_text = raw_comma_separated_part.strip()
        if not stripped_part_text:
            continue
        # Strip optional paren wrapper and weight suffix.
        bare_tag_text_to_consider = stripped_part_text
        if bare_tag_text_to_consider.startswith("(") and bare_tag_text_to_consider.endswith(")"):
            bare_tag_text_to_consider = bare_tag_text_to_consider[1:-1].strip()
            if ":" in bare_tag_text_to_consider:
                bare_tag_text_to_consider = bare_tag_text_to_consider.rsplit(":", 1)[0].strip()
        # Skip if already explicit-form (handled in the regex loop above).
        if bare_tag_text_to_consider.lower().startswith("embedding:"):
            continue
        if bare_tag_text_to_consider.lower() in embedding_lowercase_stem_to_index_entry_map:
            references_set.add(
                (bare_tag_text_to_consider, bare_tag_text_to_consider.lower(), True)
            )

    return references_set


@PromptServer.instance.routes.post(
    "/unified-conditioning-merge/validate_prompt_embeddings_sdxl"
)
async def validate_prompt_embeddings_sdxl_http_route_handler(request):
    try:
        request_body_json_payload = await request.json()
    except Exception:
        return web.json_response(
            {"messages": ["[validation request body was not valid JSON]"]},
            status=400,
        )

    prompt_texts_list = request_body_json_payload.get("prompt_texts", [])
    if not isinstance(prompt_texts_list, list):
        return web.json_response(
            {"messages": ["[validation request must include list-typed 'prompt_texts']"]},
            status=400,
        )

    # Optional field from the frontend that mirrors the runtime node toggle.
    filter_known_a1111_embedding_tags_not_installed_locally_setting_from_widget = bool(
        request_body_json_payload.get("filter_known_a1111_embedding_tags_not_installed_locally", True)
    )

    embedding_index_map = get_cached_or_build_embedding_lowercase_stem_to_index_entry_map()

    # Build the set of "known A1111 embedding names" from the file-based
    # curated list (no per-node custom override widget). Used to report
    # orphan-tag-strip actions when the corresponding toggle is on at the
    # node.
    known_a1111_embedding_names_lowercase_set_from_file = set()
    if filter_known_a1111_embedding_tags_not_installed_locally_setting_from_widget:
        try:
            from .clip_text_encode_with_cutoff_region_separation import (
                _get_cached_or_lazily_load_known_a1111_embedding_names_lowercase_set,
            )
            known_a1111_embedding_names_lowercase_set_from_file = (
                _get_cached_or_lazily_load_known_a1111_embedding_names_lowercase_set()
            )
        except Exception:
            pass

    all_embedding_references_combined_across_every_prompt_text = set()
    for one_prompt_text_string in prompt_texts_list:
        if not isinstance(one_prompt_text_string, str):
            continue
        references_in_this_text = _collect_all_embedding_references_appearing_in_one_prompt_text(
            one_prompt_text_string, embedding_index_map
        )
        all_embedding_references_combined_across_every_prompt_text |= references_in_this_text

    output_classification_message_lines = []
    sorted_references_for_stable_output = sorted(
        all_embedding_references_combined_across_every_prompt_text,
        key=lambda triple: (triple[1], triple[0]),
    )
    for (
        name_used_in_prompt,
        lowercase_stem_for_lookup,
        is_a1111_bare_style_reference,
    ) in sorted_references_for_stable_output:
        if lowercase_stem_for_lookup not in embedding_index_map:
            # File doesn't exist locally. If this is a bare tag matching the
            # known-A1111-embedding-names filter list, the runtime will strip
            # it. Otherwise it would fall through to the runtime's
            # `remove_text_for_unsupported_embeddings` path or stock encoder
            # warning. Either way: the embedding will not contribute.
            if (
                is_a1111_bare_style_reference
                and lowercase_stem_for_lookup in known_a1111_embedding_names_lowercase_set_from_file
            ):
                output_classification_message_lines.append(
                    f"Embedding {name_used_in_prompt} not installed locally, "
                    f"will be stripped (orphan A1111 tag filter)"
                )
            else:
                output_classification_message_lines.append(
                    f"Embedding {name_used_in_prompt} not found in system, will be ignored"
                )
            continue
        index_entry_for_this_embedding = embedding_index_map[lowercase_stem_for_lookup]
        if not is_embedding_file_fully_compatible_with_sdxl_based_on_its_tensor_last_dim_set(
            index_entry_for_this_embedding["tensor_last_dim_set"]
        ):
            output_classification_message_lines.append(
                f"Embedding {name_used_in_prompt} incompatible with SDXL, will be ignored"
            )

    return web.json_response({"messages": output_classification_message_lines})


@PromptServer.instance.routes.post(
    "/unified-conditioning-merge/rescan_embeddings_directory"
)
async def rescan_embeddings_directory_http_route_handler(_request):
    invalidate_cached_embedding_filename_to_dim_index()
    rebuilt_index_map = get_cached_or_build_embedding_lowercase_stem_to_index_entry_map()
    return web.json_response({"indexed_file_count": len(rebuilt_index_map)})


# ──────────────────────────────────────────────────────────────────────
# v3-specific: target-words-in-enhanced-text check.
#
# Standalone from the embedding validator above. The v3 frontend extension
# posts per-section (global_text, enhanced_text) pairs to this endpoint
# whenever the user edits a section. For each section where enhanced_text
# is non-empty AND differs from global_text, this endpoint checks whether
# each space-split word of global_text appears as a whole-word substring
# of enhanced_text. Words not found yield a warning.
#
# Word-boundary substring match (not CLIP-token-level): the v3 runtime
# encoder does the authoritative token-level match when actually building
# masks. This endpoint exists purely to surface a UI hint to the user
# while they're editing, so the simpler word-boundary check is sufficient
# and avoids needing a CLIP tokenizer here.
# ──────────────────────────────────────────────────────────────────────

V3_TARGET_WORDS_NOT_FOUND_IN_ENHANCED_WARNING_HTTP_ENDPOINT_PATH = (
    "/unified-conditioning-merge/v3_check_target_words_present_in_enhanced_text"
)


def _build_per_word_boundary_regex_pattern_for_one_target_word(target_word_string):
    """
    Returns a compiled regex that matches the target word as a whole word
    (no preceding/following word character). Used to verify the target
    appears inside the enhanced_text. Case-insensitive.
    """
    return re.compile(
        r"(?<!\w)" + re.escape(target_word_string) + r"(?!\w)",
        re.IGNORECASE,
    )


def _identify_missing_target_words_for_one_v3_section_global_in_enhanced(
    section_one_based_index, raw_global_text, raw_enhanced_text
):
    """
    Returns a list of warning strings for this section. Empty list if
    nothing to warn about.

    Premise: enhanced_text must be an EXPANDED version of global_text.
    Every word in global_text should appear somewhere inside
    enhanced_text. If a global word is MISSING from enhanced, the user
    is signaling two incompatible texts — the cutoff math derives
    targets as (enhanced_words - global_words), so a global word not
    in enhanced doesn't change target derivation directly, but it does
    indicate the user's mental model is off: global is supposed to be
    the abbreviated form, enhanced the descriptive expansion.

    Rules:
      - If enhanced_text is empty OR equals global_text (case-insensitive,
        whitespace-normalized) → this section is a passthrough, no check.
      - If global_text is empty → no check (no abbreviated form to compare).
      - Otherwise, split global_text on whitespace. For each word, verify
        it appears as a whole-word match (case-insensitive) inside
        enhanced_text. If not, emit a warning.
    """
    normalized_global_text_value = (raw_global_text or "").strip()
    normalized_enhanced_text_value = (raw_enhanced_text or "").strip()
    if not normalized_enhanced_text_value:
        return []
    if normalized_enhanced_text_value.lower() == normalized_global_text_value.lower():
        return []
    if not normalized_global_text_value:
        return []
    accumulated_warnings_for_this_section_list = []
    for one_global_word in normalized_global_text_value.split():
        if not one_global_word:
            continue
        per_word_pattern = _build_per_word_boundary_regex_pattern_for_one_target_word(
            one_global_word
        )
        if not per_word_pattern.search(normalized_enhanced_text_value):
            accumulated_warnings_for_this_section_list.append(
                f"Section {section_one_based_index}: global word "
                f"'{one_global_word}' not found in enhanced text — "
                f"enhanced should be an expanded version of global."
            )
    return accumulated_warnings_for_this_section_list


@PromptServer.instance.routes.post(
    V3_TARGET_WORDS_NOT_FOUND_IN_ENHANCED_WARNING_HTTP_ENDPOINT_PATH
)
async def v3_check_target_words_present_in_enhanced_text_http_route_handler(request):
    """
    Request body JSON:
        {"sections": [
            {"global_text": str, "enhanced_text": str},
            ...
         ]}
    Response body JSON:
        {"messages": ["Section N: target word 'foo' not found in enhanced text — ...", ...]}

    Sections are 1-indexed in messages (matches the user-visible section
    numbering in the node UI). Empty `messages` list when all sections
    pass (or are passthroughs).
    """
    try:
        request_body_json_payload = await request.json()
    except Exception:
        return web.json_response(
            {"messages": ["[v3 target-word check request body was not valid JSON]"]},
            status=400,
        )
    per_section_global_and_enhanced_pairs_list = request_body_json_payload.get("sections", [])
    if not isinstance(per_section_global_and_enhanced_pairs_list, list):
        return web.json_response(
            {"messages": ["[v3 target-word check request must include list-typed 'sections']"]},
            status=400,
        )
    accumulated_messages_across_all_sections_list = []
    for one_zero_based_section_array_index, one_section_dict in enumerate(
        per_section_global_and_enhanced_pairs_list
    ):
        if not isinstance(one_section_dict, dict):
            continue
        one_based_section_index_for_display = one_zero_based_section_array_index + 1
        global_text_value_or_empty = str(one_section_dict.get("global_text", "") or "")
        enhanced_text_value_or_empty = str(one_section_dict.get("enhanced_text", "") or "")
        per_section_warnings = _identify_missing_target_words_for_one_v3_section_global_in_enhanced(
            one_based_section_index_for_display,
            global_text_value_or_empty,
            enhanced_text_value_or_empty,
        )
        accumulated_messages_across_all_sections_list.extend(per_section_warnings)
    return web.json_response({"messages": accumulated_messages_across_all_sections_list})


# ──────────────────────────────────────────────────────────────────────
# Combined per-section validator for the detail-isolation Section node.
#
# A single endpoint that, for ONE section's (global_text, enhanced_text,
# filter_known_a1111_embedding_tags_not_installed_locally), runs BOTH
# the embedding-issue scan AND the target-word-not-in-enhanced check and
# returns a single merged list of warning messages.
#
# Different from the v1/v3 endpoints in that:
#   - Operates per-section, not on a list of section texts.
#   - Reports both classes of issue in one response so the section
#     node's bottom validator widget only needs one round-trip per edit.
# ──────────────────────────────────────────────────────────────────────

DETAIL_ISOLATION_SECTION_COMBINED_VALIDATOR_HTTP_ENDPOINT_PATH = (
    "/unified-conditioning-merge/detail_isolation_section_combined_validator"
)


@PromptServer.instance.routes.post(
    DETAIL_ISOLATION_SECTION_COMBINED_VALIDATOR_HTTP_ENDPOINT_PATH
)
async def detail_isolation_section_combined_validator_http_route_handler(request):
    """
    Request body JSON:
        {
            "global_text": str,
            "enhanced_text": str,
            "filter_known_a1111_embedding_tags_not_installed_locally": bool (default True),
        }
    Response body JSON:
        {"messages": [
            "<embedding issue line>",
            "<target word missing line>",
            ...
         ]}
    Empty `messages` list when no issues detected.
    """
    try:
        request_body_json_payload = await request.json()
    except Exception:
        return web.json_response(
            {"messages": ["[combined section validator request body was not valid JSON]"]},
            status=400,
        )
    raw_global_text_value = str(request_body_json_payload.get("global_text", "") or "")
    raw_enhanced_text_value = str(request_body_json_payload.get("enhanced_text", "") or "")
    filter_orphan_setting = bool(
        request_body_json_payload.get(
            "filter_known_a1111_embedding_tags_not_installed_locally", True
        )
    )

    combined_messages_for_this_section_list = []

    # --- (1) Target-word-not-in-enhanced check (always runs).
    target_word_check_warnings_for_this_one_section = (
        _identify_missing_target_words_for_one_v3_section_global_in_enhanced(
            1, raw_global_text_value, raw_enhanced_text_value
        )
    )
    # Section-level endpoint: strip the "Section 1:" prefix the v3 helper
    # prepends, since on a per-section node there's no section number to
    # report — the warning is implicitly about THIS section.
    for one_target_word_warning_line in target_word_check_warnings_for_this_one_section:
        line_with_section_prefix_stripped = one_target_word_warning_line
        if line_with_section_prefix_stripped.startswith("Section 1: "):
            line_with_section_prefix_stripped = line_with_section_prefix_stripped[len("Section 1: "):]
            line_with_section_prefix_stripped = (
                line_with_section_prefix_stripped[0].upper()
                + line_with_section_prefix_stripped[1:]
                if line_with_section_prefix_stripped else line_with_section_prefix_stripped
            )
        combined_messages_for_this_section_list.append(line_with_section_prefix_stripped)

    # --- (2) Embedding-issue scan over BOTH global_text and enhanced_text.
    embedding_index_map = get_cached_or_build_embedding_lowercase_stem_to_index_entry_map()
    known_a1111_embedding_names_lowercase_set_from_file = set()
    if filter_orphan_setting:
        try:
            from .clip_text_encode_with_cutoff_region_separation import (
                _get_cached_or_lazily_load_known_a1111_embedding_names_lowercase_set,
            )
            known_a1111_embedding_names_lowercase_set_from_file = (
                _get_cached_or_lazily_load_known_a1111_embedding_names_lowercase_set()
            )
        except Exception:
            pass

    accumulated_embedding_references_across_both_texts_set = set()
    for one_input_text_to_scan in (raw_global_text_value, raw_enhanced_text_value):
        accumulated_embedding_references_across_both_texts_set |= (
            _collect_all_embedding_references_appearing_in_one_prompt_text(
                one_input_text_to_scan, embedding_index_map
            )
        )
    sorted_embedding_references_for_stable_output_order = sorted(
        accumulated_embedding_references_across_both_texts_set,
        key=lambda triple: (triple[1], triple[0]),
    )
    for (
        name_used_in_prompt,
        lowercase_stem_for_lookup,
        is_a1111_bare_style_reference,
    ) in sorted_embedding_references_for_stable_output_order:
        if lowercase_stem_for_lookup not in embedding_index_map:
            if (
                is_a1111_bare_style_reference
                and lowercase_stem_for_lookup in known_a1111_embedding_names_lowercase_set_from_file
            ):
                combined_messages_for_this_section_list.append(
                    f"Embedding {name_used_in_prompt} not installed locally, "
                    f"will be stripped (orphan A1111 tag filter)"
                )
            else:
                combined_messages_for_this_section_list.append(
                    f"Embedding {name_used_in_prompt} not found in system, will be ignored"
                )
            continue
        index_entry_for_this_embedding = embedding_index_map[lowercase_stem_for_lookup]
        if not is_embedding_file_fully_compatible_with_sdxl_based_on_its_tensor_last_dim_set(
            index_entry_for_this_embedding["tensor_last_dim_set"]
        ):
            combined_messages_for_this_section_list.append(
                f"Embedding {name_used_in_prompt} incompatible with SDXL, will be ignored"
            )

    return web.json_response({"messages": combined_messages_for_this_section_list})


logging.info("unified-conditioning-merge: registered HTTP routes for realtime embedding validation + v3 target-word check + detail-isolation section combined validator.")
