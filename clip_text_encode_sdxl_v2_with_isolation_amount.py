"""
CLIP Text Encode SDXL v2 (isolation amount)

A from-scratch SDXL text encoder using a simpler split-by-isolation-amount
model than the v1 cutoff-paper algorithm. See docs/v2_text_encoder_pseudocode.md
for the full design rationale.

Per-region widgets (default count = 2, range 0..32):
  region_N_text                                  STRING multiline
  region_N_weight                                FLOAT -10..10, default 1.0
  region_N_isolation_amount                      FLOAT 0..1, default 1.0
  region_N_clip_l_strength                       FLOAT -10..10, default 1.0
  region_N_clip_g_strength                       FLOAT -10..10, default 1.0
  region_N_weight_from_other_isolated_regions    FLOAT -10..10, default 0.0

Endpoint categorization rule:
  isolation == 0.0 → region appears ONLY in the global stack (not in
                     isolated list even at weight 0).
  isolation == 1.0 → region appears ONLY in the isolated list.
  0 < isolation < 1 → region appears in BOTH lists with fractional weights.

Per-stream strength of 0 → region excluded ENTIRELY from that CLIP stream's
                           prompt construction (avoids the `(tag:0)` paradox).

Output: single composite CONDITIONING entry, position-mapped per the
canonical comma-joined sequence of all region texts in declaration order.
"""

import logging
import re

import torch

import nodes

from .cutoff_per_stream_isolation import (
    _get_per_stream_clip_text_encoder_model_and_tokenizer,
    _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad,
    _find_all_sublist_match_start_positions_within_superlist,
)

# Shared feature helpers ported from v1 (clip_text_encode_with_cutoff_region_separation.py)
# — A1111-style embedding rewriting, unsupported-embedding stripping, orphan-tag
# filter, shape-mismatch warnings, SDXL zoom/crop metadata builder, latent W/H
# resolver, and the upscaled-conditioning multiplier constants.
from .clip_text_encode_with_cutoff_region_separation import (
    _rewrite_a1111_style_bare_embedding_tags_to_comfyui_embedding_prefix_form,
    _strip_unsupported_embedding_references_from_prompt_text,
    _strip_orphan_a1111_bare_tags_matching_known_names_list_but_not_installed_locally,
    _get_cached_or_lazily_load_known_a1111_embedding_names_lowercase_set,
    _build_lookup_of_available_embedding_stems_to_their_filename_list,
    _emit_stock_style_shape_mismatch_warning_for_each_embedding_in_text_that_will_not_match_its_clip_stream_dim,
    _compute_sdxl_size_and_crop_metadata_fields,
    _resolve_target_image_width_and_height_from_optional_latent_or_defaults,
    _clamp_numeric_value_inclusive,
    UPSCALED_CONDITIONING_MULTIPLIER_DEFAULT_VALUE,
    UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE,
    ZOOM_DEFAULT_VALUE,
    ZOOM_MINIMUM_VALUE,
    ZOOM_MAXIMUM_VALUE,
    OFFSET_DEFAULT_VALUE,
    OFFSET_MINIMUM_VALUE,
    OFFSET_MAXIMUM_VALUE,
)


def _parse_custom_embedding_names_string_into_lowercase_set(custom_names_input_text_string):
    """
    Parse the v2 node's `custom_embedding_names_to_strip` widget value into a
    lowercase set, mirroring the same-named helper in v1's cutoff module.
    Lines and/or comma-separated names; '#' lines are comments.
    """
    parsed_lowercase_names_set = set()
    if not custom_names_input_text_string:
        return parsed_lowercase_names_set
    for one_raw_line in custom_names_input_text_string.splitlines():
        stripped_line_text = one_raw_line.strip()
        if not stripped_line_text or stripped_line_text.startswith("#"):
            continue
        for one_comma_separated_name in stripped_line_text.split(","):
            stripped_name = one_comma_separated_name.strip()
            if stripped_name:
                parsed_lowercase_names_set.add(stripped_name.lower())
    return parsed_lowercase_names_set


def _apply_v1_style_prompt_text_transforms_to_one_region_text(
    region_text_string,
    clip_object,
    support_a1111_style_embedding_text_setting,
    remove_text_for_unsupported_embeddings_setting,
    filter_known_a1111_embedding_tags_not_installed_locally_setting,
    custom_embedding_names_to_strip_setting,
):
    """
    Applies the same pre-encode text transforms v1 runs on each group's
    prompt text, but to a single region's text in isolation. Order matches
    v1:
      1. Orphan-A1111-tag filter (drops uninstalled bare-name tags that
         appear in the curated + custom names list).
      2. A1111-style bare-tag rewrite (installed-locally bare tags →
         `embedding:NAME`).
      3. Unsupported-embedding text strip (drops `embedding:NAME` refs
         whose tensor doesn't fit any CLIP stream's expected dim).
      4. Shape-mismatch warning logging (does not mutate text).
    """
    working_text = region_text_string or ""
    if filter_known_a1111_embedding_tags_not_installed_locally_setting:
        available_embedding_lowercase_stem_to_filenames_map = (
            _build_lookup_of_available_embedding_stems_to_their_filename_list()
        )
        custom_additional_names_lowercase_set = (
            _parse_custom_embedding_names_string_into_lowercase_set(
                custom_embedding_names_to_strip_setting
            )
        )
        working_text = (
            _strip_orphan_a1111_bare_tags_matching_known_names_list_but_not_installed_locally(
                working_text,
                available_embedding_lowercase_stem_to_filenames_map,
                additional_custom_names_lowercase_set=custom_additional_names_lowercase_set,
            )
        )
    if support_a1111_style_embedding_text_setting:
        working_text = _rewrite_a1111_style_bare_embedding_tags_to_comfyui_embedding_prefix_form(
            working_text
        )
    if remove_text_for_unsupported_embeddings_setting:
        working_text = _strip_unsupported_embedding_references_from_prompt_text(
            working_text, clip_object
        )
    _emit_stock_style_shape_mismatch_warning_for_each_embedding_in_text_that_will_not_match_its_clip_stream_dim(
        clip_object, working_text
    )
    return working_text


# Constants
MAX_REGION_COUNT_SUPPORTED = 32
DEFAULT_REGION_COUNT_VALUE = 2

SDXL_CLIP_L_EXPECTED_EMBEDDING_DIM = 768
SDXL_CLIP_G_EXPECTED_EMBEDDING_DIM = 1280


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _collect_active_region_descriptors_from_kwargs_in_declaration_order(
    kwargs_with_per_region_widget_values, region_count_active_setting_value
):
    """
    Pulls the per-region widget values out of the flat **kwargs that
    ComfyUI hands to the encode method and returns a list of region
    descriptor dicts (only for region indices 1..region_count_active).
    Regions with empty text after stripping whitespace are skipped.
    """
    collected_region_descriptor_list_in_declaration_order = []
    for one_based_region_index in range(1, int(region_count_active_setting_value) + 1):
        text_for_this_region_from_widget = str(
            kwargs_with_per_region_widget_values.get(
                f"region_{one_based_region_index}_text", ""
            ) or ""
        )
        if not text_for_this_region_from_widget.strip():
            continue
        collected_region_descriptor_list_in_declaration_order.append({
            "id": one_based_region_index,
            "text": text_for_this_region_from_widget.strip(),
            "weight": float(
                kwargs_with_per_region_widget_values.get(
                    f"region_{one_based_region_index}_weight", 1.0
                )
            ),
            "isolation": float(
                kwargs_with_per_region_widget_values.get(
                    f"region_{one_based_region_index}_isolation_amount", 1.0
                )
            ),
            "clip_l_strength": float(
                kwargs_with_per_region_widget_values.get(
                    f"region_{one_based_region_index}_clip_l_strength", 1.0
                )
            ),
            "clip_g_strength": float(
                kwargs_with_per_region_widget_values.get(
                    f"region_{one_based_region_index}_clip_g_strength", 1.0
                )
            ),
            "weight_from_other_isolated_regions": float(
                kwargs_with_per_region_widget_values.get(
                    f"region_{one_based_region_index}_weight_from_other_isolated_regions",
                    0.0,
                )
            ),
        })
    return collected_region_descriptor_list_in_declaration_order


def _select_stream_strength_value_for_region_on_one_stream(region_descriptor, stream_key_l_or_g):
    if stream_key_l_or_g == "l":
        return region_descriptor["clip_l_strength"]
    return region_descriptor["clip_g_strength"]


def _wrap_text_with_clip_weight_syntax(text_string, numerical_weight_value):
    """
    Returns the CLIP `(text:weight)` form. Weight is rounded to 4 decimals
    so absurdly long floats don't bloat the prompt string. Weight of
    exactly 1.0 still gets the wrapper for parsing uniformity.
    """
    rounded_weight_for_clip_syntax = round(float(numerical_weight_value), 4)
    return f"({text_string}:{rounded_weight_for_clip_syntax})"


def _build_global_stack_prompt_parts_for_one_stream(
    regions_in_global_stack_for_this_call, stream_key_l_or_g
):
    """
    Per the categorization rule, `regions_in_global_stack_for_this_call`
    should already exclude any region with isolation == 1.0.
    """
    global_stack_text_parts_in_declaration_order = []
    for region_descriptor in regions_in_global_stack_for_this_call:
        per_stream_strength_value = _select_stream_strength_value_for_region_on_one_stream(
            region_descriptor, stream_key_l_or_g
        )
        if per_stream_strength_value == 0:
            continue
        global_portion_contribution_weight = (
            region_descriptor["weight"]
            * (1.0 - region_descriptor["isolation"])
            * per_stream_strength_value
        )
        global_stack_text_parts_in_declaration_order.append(
            _wrap_text_with_clip_weight_syntax(
                region_descriptor["text"], global_portion_contribution_weight
            )
        )
    return global_stack_text_parts_in_declaration_order


def _build_per_region_full_prompt_text_for_one_region_on_one_stream(
    region_being_encoded,
    global_stack_text_parts_for_this_stream,
    regions_in_isolated_list_for_this_call,
    stream_key_l_or_g,
):
    """
    Builds the prompt that THIS region will be encoded against for THIS
    stream:
        global stack
      + own isolated portion (if isolation > 0)
      + each OTHER isolated region pulled in at scaled weight (if
        weight_from_other != 0)

    Returns None if this region's per-stream strength is 0 (the region is
    fully excluded from this stream).
    """
    per_stream_strength_value_for_this_region = (
        _select_stream_strength_value_for_region_on_one_stream(
            region_being_encoded, stream_key_l_or_g
        )
    )
    if per_stream_strength_value_for_this_region == 0:
        return None

    per_region_prompt_parts_in_layered_order = list(global_stack_text_parts_for_this_stream)

    if region_being_encoded["isolation"] > 0:
        own_isolated_portion_weight = (
            region_being_encoded["weight"]
            * region_being_encoded["isolation"]
            * per_stream_strength_value_for_this_region
        )
        per_region_prompt_parts_in_layered_order.append(
            _wrap_text_with_clip_weight_syntax(
                region_being_encoded["text"], own_isolated_portion_weight
            )
        )

    if region_being_encoded["weight_from_other_isolated_regions"] != 0:
        for other_isolated_region in regions_in_isolated_list_for_this_call:
            if other_isolated_region["id"] == region_being_encoded["id"]:
                continue
            other_stream_strength_value = (
                _select_stream_strength_value_for_region_on_one_stream(
                    other_isolated_region, stream_key_l_or_g
                )
            )
            if other_stream_strength_value == 0:
                continue
            pulled_in_weight_from_this_other_region = (
                region_being_encoded["weight_from_other_isolated_regions"]
                * other_isolated_region["weight"]
                * other_isolated_region["isolation"]
                * other_stream_strength_value
            )
            per_region_prompt_parts_in_layered_order.append(
                _wrap_text_with_clip_weight_syntax(
                    other_isolated_region["text"], pulled_in_weight_from_this_other_region
                )
            )

    return ", ".join(per_region_prompt_parts_in_layered_order)


def _encode_tokens_per_stream_returning_embedding_tensor_and_pooled_or_none(
    clip_object, tokens_per_chunk_for_one_stream, stream_key_l_or_g
):
    """
    Calls the per-stream encoder model directly so we can encode a single
    stream's chunk list without going through SDXLClipModel's joint path.
    Returns (embedding_tensor, pooled_or_none) — pooled_or_none is the
    pooled output if the stream has one (typically only G for SDXL).
    """
    per_stream_encoder_model, _per_stream_tokenizer = (
        _get_per_stream_clip_text_encoder_model_and_tokenizer(clip_object, stream_key_l_or_g)
    )
    embedding_tensor_output_for_this_stream, pooled_output_for_this_stream = (
        per_stream_encoder_model.encode_token_weights(tokens_per_chunk_for_one_stream)
    )
    return embedding_tensor_output_for_this_stream, pooled_output_for_this_stream


def _find_canonical_position_range_for_region_text_within_canonical_content_token_ids(
    canonical_content_token_ids_flat_list, region_target_text_content_token_ids_flat_list
):
    """
    Returns (start_in_content, end_in_content) for the first match of
    region_target tokens within canonical_content tokens, or None if no
    match is found.
    """
    if not region_target_text_content_token_ids_flat_list:
        return None
    all_match_start_positions = _find_all_sublist_match_start_positions_within_superlist(
        canonical_content_token_ids_flat_list,
        region_target_text_content_token_ids_flat_list,
    )
    if not all_match_start_positions:
        return None
    first_match_start = all_match_start_positions[0]
    first_match_end_exclusive = first_match_start + len(region_target_text_content_token_ids_flat_list)
    return (first_match_start, first_match_end_exclusive)


def _convert_content_position_range_to_full_chunk_position_range(
    content_position_start_inclusive,
    content_position_end_exclusive,
    chunk_size_with_markers=77,
    chunk_content_size_excluding_markers=75,
):
    """
    Content positions (skipping start/end markers per chunk) map to full
    chunk positions by adding 1-per-chunk offsets for each chunk's start
    marker, then 2-per-chunk offsets for skipped end markers... simpler
    formula: for content position p, the full chunk position is
       chunk_index * 77 + 1 + position_within_chunk_content_window
    where chunk_index = p // 75 and position_within = p % 75.
    """
    chunk_index_for_start_position = (
        content_position_start_inclusive // chunk_content_size_excluding_markers
    )
    position_within_chunk_for_start = (
        content_position_start_inclusive % chunk_content_size_excluding_markers
    )
    full_chunk_position_for_start = (
        chunk_index_for_start_position * chunk_size_with_markers
        + 1
        + position_within_chunk_for_start
    )

    chunk_index_for_end_position = (
        (content_position_end_exclusive - 1) // chunk_content_size_excluding_markers
    )
    position_within_chunk_for_end_minus_one = (
        (content_position_end_exclusive - 1) % chunk_content_size_excluding_markers
    )
    full_chunk_position_for_end_exclusive = (
        chunk_index_for_end_position * chunk_size_with_markers
        + 1
        + position_within_chunk_for_end_minus_one
        + 1
    )

    return (full_chunk_position_for_start, full_chunk_position_for_end_exclusive)


# ──────────────────────────────────────────────────────────────────────
# Main encoder
# ──────────────────────────────────────────────────────────────────────

def _encode_v2_for_one_stream_returning_final_embedding_and_pooled(
    clip_object,
    active_region_descriptors_list,
    stream_key_l_or_g,
):
    # Categorize regions per the endpoint exclusion rule
    regions_in_global_stack_for_this_stream_call = [
        R for R in active_region_descriptors_list if R["isolation"] < 1.0
    ]
    regions_in_isolated_list_for_this_stream_call = [
        R for R in active_region_descriptors_list if R["isolation"] > 0.0
    ]

    # Build the global stack text for this stream
    global_stack_text_parts_for_this_stream = _build_global_stack_prompt_parts_for_one_stream(
        regions_in_global_stack_for_this_stream_call, stream_key_l_or_g
    )

    # Per-region encoding pass: build each region's custom prompt and encode
    per_region_id_to_encoded_embedding_for_this_stream = {}
    per_region_id_to_tokens_used_for_its_encoding = {}
    for region_descriptor in active_region_descriptors_list:
        per_region_full_prompt_text = _build_per_region_full_prompt_text_for_one_region_on_one_stream(
            region_descriptor,
            global_stack_text_parts_for_this_stream,
            regions_in_isolated_list_for_this_stream_call,
            stream_key_l_or_g,
        )
        if per_region_full_prompt_text is None:
            per_region_id_to_encoded_embedding_for_this_stream[region_descriptor["id"]] = None
            per_region_id_to_tokens_used_for_its_encoding[region_descriptor["id"]] = None
            continue

        tokens_for_this_region_on_this_stream = clip_object.tokenize(
            per_region_full_prompt_text
        )[stream_key_l_or_g]
        encoded_embedding_for_this_region_on_this_stream, _unused_pooled = (
            _encode_tokens_per_stream_returning_embedding_tensor_and_pooled_or_none(
                clip_object, tokens_for_this_region_on_this_stream, stream_key_l_or_g
            )
        )
        per_region_id_to_encoded_embedding_for_this_stream[region_descriptor["id"]] = (
            encoded_embedding_for_this_region_on_this_stream
        )
        per_region_id_to_tokens_used_for_its_encoding[region_descriptor["id"]] = (
            tokens_for_this_region_on_this_stream
        )

    # Canonical sequence: raw comma-joined text of all regions in declaration order
    canonical_raw_joined_text = ", ".join(
        R["text"] for R in active_region_descriptors_list
    )
    canonical_tokens_for_this_stream = clip_object.tokenize(canonical_raw_joined_text)[stream_key_l_or_g]
    canonical_encoded_embedding_for_this_stream, canonical_pooled_output_for_this_stream = (
        _encode_tokens_per_stream_returning_embedding_tensor_and_pooled_or_none(
            clip_object, canonical_tokens_for_this_stream, stream_key_l_or_g
        )
    )

    _per_stream_encoder, per_stream_tokenizer = _get_per_stream_clip_text_encoder_model_and_tokenizer(
        clip_object, stream_key_l_or_g
    )
    end_token_id_for_this_stream = per_stream_tokenizer.end_token

    canonical_content_token_ids = (
        _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad(
            canonical_tokens_for_this_stream, end_token_id_for_this_stream
        )
    )

    # Determine each region's content-position range within canonical
    per_region_id_to_canonical_content_position_range = {}
    per_region_id_to_own_content_position_range_in_own_encoding = {}
    for region_descriptor in active_region_descriptors_list:
        per_stream_strength = _select_stream_strength_value_for_region_on_one_stream(
            region_descriptor, stream_key_l_or_g
        )
        if per_stream_strength == 0:
            per_region_id_to_canonical_content_position_range[region_descriptor["id"]] = None
            per_region_id_to_own_content_position_range_in_own_encoding[region_descriptor["id"]] = None
            continue

        region_own_text_tokens = clip_object.tokenize(region_descriptor["text"])[stream_key_l_or_g]
        region_own_text_content_ids = (
            _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad(
                region_own_text_tokens, end_token_id_for_this_stream
            )
        )

        canonical_position_range = (
            _find_canonical_position_range_for_region_text_within_canonical_content_token_ids(
                canonical_content_token_ids, region_own_text_content_ids
            )
        )
        per_region_id_to_canonical_content_position_range[region_descriptor["id"]] = canonical_position_range

        # Where does the region's own text live within its own per-region encoding's tokens?
        own_per_region_tokens = per_region_id_to_tokens_used_for_its_encoding[region_descriptor["id"]]
        if own_per_region_tokens is None:
            per_region_id_to_own_content_position_range_in_own_encoding[region_descriptor["id"]] = None
            continue
        own_per_region_content_ids = (
            _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad(
                own_per_region_tokens, end_token_id_for_this_stream
            )
        )
        own_text_position_range_in_own_encoding = (
            _find_canonical_position_range_for_region_text_within_canonical_content_token_ids(
                own_per_region_content_ids, region_own_text_content_ids
            )
        )
        per_region_id_to_own_content_position_range_in_own_encoding[region_descriptor["id"]] = (
            own_text_position_range_in_own_encoding
        )

    # Compose final embedding: start from canonical, overlay per-region encodings at their canonical positions
    final_embedding_tensor_for_this_stream = canonical_encoded_embedding_for_this_stream.clone()
    canonical_full_chunk_token_count = canonical_encoded_embedding_for_this_stream.shape[1]

    for region_descriptor in active_region_descriptors_list:
        region_encoding = per_region_id_to_encoded_embedding_for_this_stream[region_descriptor["id"]]
        canonical_content_position_range = per_region_id_to_canonical_content_position_range[
            region_descriptor["id"]
        ]
        own_content_position_range_in_own_encoding = (
            per_region_id_to_own_content_position_range_in_own_encoding[region_descriptor["id"]]
        )
        if (
            region_encoding is None
            or canonical_content_position_range is None
            or own_content_position_range_in_own_encoding is None
        ):
            continue

        # Convert content positions to full chunk positions
        canonical_full_position_range = _convert_content_position_range_to_full_chunk_position_range(
            canonical_content_position_range[0], canonical_content_position_range[1]
        )
        own_full_position_range = _convert_content_position_range_to_full_chunk_position_range(
            own_content_position_range_in_own_encoding[0],
            own_content_position_range_in_own_encoding[1],
        )

        canonical_full_start, canonical_full_end = canonical_full_position_range
        own_full_start, own_full_end = own_full_position_range

        canonical_span_length = canonical_full_end - canonical_full_start
        own_span_length = own_full_end - own_full_start
        clamped_span_length = min(canonical_span_length, own_span_length)

        # Bounds-check against the tensor sizes
        canonical_full_end_clamped = min(
            canonical_full_start + clamped_span_length, canonical_full_chunk_token_count
        )
        own_full_end_clamped = min(
            own_full_start + clamped_span_length, region_encoding.shape[1]
        )
        actual_copy_span_length = min(
            canonical_full_end_clamped - canonical_full_start,
            own_full_end_clamped - own_full_start,
        )
        if actual_copy_span_length <= 0:
            continue

        final_embedding_tensor_for_this_stream[
            :, canonical_full_start : canonical_full_start + actual_copy_span_length, :
        ] = region_encoding[
            :, own_full_start : own_full_start + actual_copy_span_length, :
        ]

    return final_embedding_tensor_for_this_stream, canonical_pooled_output_for_this_stream


# ──────────────────────────────────────────────────────────────────────
# Node class
# ──────────────────────────────────────────────────────────────────────

class CLIPTextEncodeSDXLV2WithIsolationAmount:
    @classmethod
    def INPUT_TYPES(cls):
        required_inputs_dict = {
            "clip": ("CLIP",),
            "upscaled_conditioning_multiplier": ("FLOAT", {
                "default": UPSCALED_CONDITIONING_MULTIPLIER_DEFAULT_VALUE,
                "min": UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE,
                "step": 0.01,
            }),
            "region_count": ("INT", {
                "default": DEFAULT_REGION_COUNT_VALUE,
                "min": 0,
                "max": MAX_REGION_COUNT_SUPPORTED,
                "step": 1,
            }),
            "support_a1111_style_embedding_text": ("BOOLEAN", {"default": True}),
            "remove_text_for_unsupported_embeddings": ("BOOLEAN", {"default": True}),
            "filter_known_a1111_embedding_tags_not_installed_locally": ("BOOLEAN", {
                "default": True,
                "tooltip": (
                    "List can be modified in custom node folder: "
                    "known_a1111_embedding_names_to_filter_when_not_installed_locally.txt"
                ),
            }),
            "custom_embedding_names_to_strip": ("STRING", {
                "multiline": True,
                "default": "",
                "placeholder": "One name per line (or comma-separated). Added to the file-based known-names filter.",
            }),
            # Zoom-effect group (zoom + offset_x + offset_y). A canvas-drawn
            # header label is inserted above these by the v2 frontend
            # extension web/clip_text_encode_sdxl_v2_with_isolation_amount.js.
            "zoom": ("FLOAT", {
                "default": ZOOM_DEFAULT_VALUE,
                "min": ZOOM_MINIMUM_VALUE,
                "max": ZOOM_MAXIMUM_VALUE,
                "step": 0.01,
            }),
            "offset_x": ("FLOAT", {
                "default": OFFSET_DEFAULT_VALUE,
                "min": OFFSET_MINIMUM_VALUE,
                "max": OFFSET_MAXIMUM_VALUE,
                "step": 0.01,
            }),
            "offset_y": ("FLOAT", {
                "default": OFFSET_DEFAULT_VALUE,
                "min": OFFSET_MINIMUM_VALUE,
                "max": OFFSET_MAXIMUM_VALUE,
                "step": 0.01,
            }),
        }
        for one_based_region_index_for_declaration in range(1, MAX_REGION_COUNT_SUPPORTED + 1):
            required_inputs_dict[f"region_{one_based_region_index_for_declaration}_text"] = (
                "STRING", {"multiline": True, "default": ""},
            )
            required_inputs_dict[f"region_{one_based_region_index_for_declaration}_weight"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
            required_inputs_dict[f"region_{one_based_region_index_for_declaration}_isolation_amount"] = (
                "FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
            )
            required_inputs_dict[f"region_{one_based_region_index_for_declaration}_clip_l_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
            required_inputs_dict[f"region_{one_based_region_index_for_declaration}_clip_g_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
            required_inputs_dict[
                f"region_{one_based_region_index_for_declaration}_weight_from_other_isolated_regions"
            ] = (
                "FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
        return {
            "required": required_inputs_dict,
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "upscaled_conditioning", "reference_full_prompt")
    FUNCTION = "encode_v2"
    CATEGORY = "unified-conditioning-merge"

    def encode_v2(
        self,
        clip,
        upscaled_conditioning_multiplier,
        region_count,
        support_a1111_style_embedding_text,
        remove_text_for_unsupported_embeddings,
        filter_known_a1111_embedding_tags_not_installed_locally,
        custom_embedding_names_to_strip,
        zoom,
        offset_x,
        offset_y,
        latent=None,
        **kwargs_with_per_region_widget_values,
    ):
        active_region_descriptors_list = (
            _collect_active_region_descriptors_from_kwargs_in_declaration_order(
                kwargs_with_per_region_widget_values, region_count
            )
        )

        # Apply v1's per-prompt text transforms to each active region BEFORE
        # any encoding work. Each region is treated as an independent prompt
        # text for transform purposes: orphan-A1111-tag filter, A1111 bare-tag
        # rewrite, unsupported-embedding strip, shape-mismatch warning logging.
        for region_descriptor_to_mutate in active_region_descriptors_list:
            region_descriptor_to_mutate["text"] = (
                _apply_v1_style_prompt_text_transforms_to_one_region_text(
                    region_descriptor_to_mutate["text"],
                    clip,
                    bool(support_a1111_style_embedding_text),
                    bool(remove_text_for_unsupported_embeddings),
                    bool(filter_known_a1111_embedding_tags_not_installed_locally),
                    str(custom_embedding_names_to_strip or ""),
                )
            )
        # Some regions may now have empty text (entire body was stripped); drop them.
        active_region_descriptors_list = [
            R for R in active_region_descriptors_list if (R["text"] or "").strip()
        ]

        # Resolve SDXL geometry: target W/H from latent if provided, else 1024².
        # Build both primary and (multiplied) upscaled metadata dicts.
        primary_target_image_width, primary_target_image_height = (
            _resolve_target_image_width_and_height_from_optional_latent_or_defaults(latent)
        )
        conditioning_upscale_factor_clamped = max(
            UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE,
            float(upscaled_conditioning_multiplier),
        )
        upscaled_target_image_width = int(round(primary_target_image_width * conditioning_upscale_factor_clamped))
        upscaled_target_image_height = int(round(primary_target_image_height * conditioning_upscale_factor_clamped))

        zoom_factor_clamped = max(ZOOM_MINIMUM_VALUE, float(zoom))
        offset_x_clamped = _clamp_numeric_value_inclusive(
            float(offset_x), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE
        )
        offset_y_clamped = _clamp_numeric_value_inclusive(
            float(offset_y), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE
        )
        primary_sdxl_size_and_crop_metadata_fields = _compute_sdxl_size_and_crop_metadata_fields(
            primary_target_image_width, primary_target_image_height,
            zoom_factor_clamped, offset_x_clamped, offset_y_clamped,
        )
        upscaled_targets_differ_from_primary = (
            upscaled_target_image_width != primary_target_image_width
            or upscaled_target_image_height != primary_target_image_height
        )
        if upscaled_targets_differ_from_primary:
            upscaled_sdxl_size_and_crop_metadata_fields = _compute_sdxl_size_and_crop_metadata_fields(
                upscaled_target_image_width, upscaled_target_image_height,
                zoom_factor_clamped, offset_x_clamped, offset_y_clamped,
            )
        else:
            upscaled_sdxl_size_and_crop_metadata_fields = primary_sdxl_size_and_crop_metadata_fields

        if not active_region_descriptors_list:
            # Encode empty prompt as fallback so the workflow can still run.
            empty_prompt_tokens = clip.tokenize("")
            primary_empty_cond_list = clip.encode_from_tokens_scheduled(
                empty_prompt_tokens, add_dict=primary_sdxl_size_and_crop_metadata_fields,
            )
            upscaled_empty_cond_list = clip.encode_from_tokens_scheduled(
                empty_prompt_tokens, add_dict=upscaled_sdxl_size_and_crop_metadata_fields,
            )
            return (primary_empty_cond_list, upscaled_empty_cond_list, "")

        # Encode per stream independently with the v2 isolation-amount math.
        final_per_stream_embedding_tensor = {}
        final_per_stream_pooled_output_or_none = {}
        for stream_key in ("l", "g"):
            (
                embedding_tensor,
                pooled_or_none,
            ) = _encode_v2_for_one_stream_returning_final_embedding_and_pooled(
                clip, active_region_descriptors_list, stream_key
            )
            final_per_stream_embedding_tensor[stream_key] = embedding_tensor
            final_per_stream_pooled_output_or_none[stream_key] = pooled_or_none

        # SDXL combine: cat L (768) + G (1280) along last dim. Truncate to shared seq length.
        shared_sequence_length = min(
            final_per_stream_embedding_tensor["l"].shape[1],
            final_per_stream_embedding_tensor["g"].shape[1],
        )
        sdxl_combined_token_embedding_tensor = torch.cat(
            [
                final_per_stream_embedding_tensor["l"][:, :shared_sequence_length],
                final_per_stream_embedding_tensor["g"][:, :shared_sequence_length],
            ],
            dim=-1,
        )

        # Build two CONDITIONING outputs sharing the token tensor but with
        # distinct SDXL metadata (primary target W/H vs upscaled target W/H).
        primary_metadata_dict = dict(primary_sdxl_size_and_crop_metadata_fields)
        upscaled_metadata_dict = dict(upscaled_sdxl_size_and_crop_metadata_fields)
        if final_per_stream_pooled_output_or_none.get("g") is not None:
            primary_metadata_dict["pooled_output"] = final_per_stream_pooled_output_or_none["g"]
            upscaled_metadata_dict["pooled_output"] = final_per_stream_pooled_output_or_none["g"]

        reference_full_prompt_text_for_output = ", ".join(
            R["text"] for R in active_region_descriptors_list
        )
        return (
            [[sdxl_combined_token_embedding_tensor, primary_metadata_dict]],
            [[sdxl_combined_token_embedding_tensor, upscaled_metadata_dict]],
            reference_full_prompt_text_for_output,
        )
