"""
CLIP Text Encode SDXL Enhanced (Inline Tags)

A simplified single-text-field node that uses inline XML-style tags to
mark regions and their detail-masking targets. The natural prompt is
just the typed text with tags removed — making prompts copy-pasteable
to/from any other CLIP encoder.

Tag syntax (both opening and closing tags required, case-insensitive):

  <REGION>... region body ...</REGION>
      Marks a contiguous span as a cutoff-style region. The body
      becomes part of the natural base prompt verbatim (tags stripped).
      The model "reads" exactly what's between the tags.

  <DETAIL>... detail text ...</DETAIL>   (used INSIDE a <REGION>)
      Marks words within the surrounding region that should be MASKED
      from OTHER regions' encoding passes — i.e., these are the
      "distinctive" details that shouldn't bleed into other regions.
      Whitespace-split into individual target words; each word
      independently masked. Multi-word details get split into each
      separate word target (cutoff's standard behavior).

  Text OUTSIDE any <REGION> block is a passthrough — it appears in
  the base prompt verbatim but is not a region (no cutoff overlay,
  no targets).

Example:

  <REGION>a <DETAIL>gnarled old</DETAIL> tree <DETAIL>with a wicked face</DETAIL></REGION>, <REGION>a bird <DETAIL>massive, red eyes, flying</DETAIL></REGION>

  → Base prompt encoded by CLIP:
      "a gnarled old tree with a wicked face, a bird massive, red eyes, flying"

  → Region 1 ("a gnarled old tree with a wicked face")
      target words to mask in other regions:
          gnarled, old, with, a, wicked, face

  → Region 2 ("a bird massive, red eyes, flying")
      target words to mask in other regions:
          massive,, red, eyes,, flying
      (punctuation kept as part of whitespace-split tokens — same
      behavior as BNK's target_text.split(" "))

If you paste this same text into stock CLIPTextEncode, you get the
same plain prompt because the angle-bracket tags are valid in CLIP
attention markup but produce no meaningful effect (they don't match
A1111/ComfyUI weight syntax). The model just reads the text content.

Outputs match v1/v3:
    conditioning            (CONDITIONING, primary SDXL metadata)
    upscaled_conditioning   (CONDITIONING, same tokens, upscaled metadata)
    reference_full_prompt   (STRING, the cleaned base prompt the model encoded)
"""

import logging
import re

from .clip_text_encode_sdxl_v3_global_and_enhanced import (
    apply_v3_per_text_transforms_to_one_text_string,
    encode_active_v3_sections_into_one_sdxl_conditioning_entry,
    build_per_stream_base_prompt_text_and_per_section_base_fragment_list,
    build_plain_text_reference_prompt_without_clip_weight_wrapping_for_display,
    compute_sdxl_size_and_crop_metadata_fields,
    resolve_target_image_width_and_height_from_optional_latent_or_defaults,
    clamp_numeric_value_inclusive,
)
from .clip_text_encode_with_cutoff_region_separation import (
    UPSCALED_CONDITIONING_MULTIPLIER_DEFAULT_VALUE,
    UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE,
    ZOOM_DEFAULT_VALUE,
    ZOOM_MINIMUM_VALUE,
    ZOOM_MAXIMUM_VALUE,
    OFFSET_DEFAULT_VALUE,
    OFFSET_MINIMUM_VALUE,
    OFFSET_MAXIMUM_VALUE,
)


# Capture <REGION>...</REGION> blocks (case-insensitive, dotall so the
# body can span newlines). The opening tag MAY include optional
# attributes like `start_timestep:0.0 end_timestep:0.5`.
REGION_BLOCK_WITH_OPTIONAL_ATTRS_REGEX_PATTERN = re.compile(
    r"<\s*REGION\b(?P<attrs_block_inside_opening_tag>[^>]*?)>"
    r"(?P<region_body_between_tags>.*?)"
    r"<\s*/\s*REGION\s*>",
    re.IGNORECASE | re.DOTALL,
)
# Per-attribute pattern: `key:value` separated by whitespace inside
# the opening tag's attrs block. value matches up to next whitespace.
REGION_ONE_ATTRIBUTE_KEY_VALUE_PAIR_REGEX_PATTERN = re.compile(
    r"(?P<attr_key>\w+)\s*:\s*(?P<attr_value>[\w.+\-]+)"
)

# Capture <DETAIL>...</DETAIL> blocks inside a region body.
DETAIL_BLOCK_REGEX_PATTERN = re.compile(
    r"<\s*DETAIL\s*>(.*?)<\s*/\s*DETAIL\s*>", re.IGNORECASE | re.DOTALL
)

DEFAULT_TIMESTEP_RANGE_START_VALUE = 0.0
DEFAULT_TIMESTEP_RANGE_END_VALUE = 1.0
FLOAT_EPSILON_FOR_TIMESTEP_BREAKPOINT_DEDUPE = 1e-9


def _parse_optional_region_attributes_block_into_timestep_range_dict(attrs_block_text_inside_opening_tag):
    """
    Parses the attribute portion inside a `<REGION ...>` opening tag.
    Returns a dict of attr_key -> attr_value (both strings). Currently
    we only USE `start_timestep` and `end_timestep`; unknown keys are
    preserved in the dict for future extensions but ignored downstream.

    Examples:
        ""                                          -> {}
        " start_timestep:0.0 end_timestep:0.5 "     -> {"start_timestep":"0.0","end_timestep":"0.5"}
    """
    parsed_attribute_dict = {}
    if not attrs_block_text_inside_opening_tag:
        return parsed_attribute_dict
    for one_attribute_match in REGION_ONE_ATTRIBUTE_KEY_VALUE_PAIR_REGEX_PATTERN.finditer(
        attrs_block_text_inside_opening_tag
    ):
        parsed_attribute_dict[one_attribute_match.group("attr_key").lower()] = (
            one_attribute_match.group("attr_value")
        )
    return parsed_attribute_dict


def _parse_one_timestep_attribute_value_to_clamped_float(
    raw_attribute_value_or_none, default_value_when_missing_or_invalid
):
    """
    Parse a single timestep attribute string into a float, clamped to
    [0.0, 1.0]. Falls back to default when missing or unparseable.
    """
    if raw_attribute_value_or_none is None:
        return float(default_value_when_missing_or_invalid)
    try:
        parsed_value_as_float = float(raw_attribute_value_or_none)
    except (TypeError, ValueError):
        return float(default_value_when_missing_or_invalid)
    if parsed_value_as_float < 0.0:
        parsed_value_as_float = 0.0
    if parsed_value_as_float > 1.0:
        parsed_value_as_float = 1.0
    return parsed_value_as_float


def _strip_detail_tags_keeping_inline_content_from_one_region_body(region_body_text_with_detail_tags):
    """
    Returns the region body with <DETAIL>...</DETAIL> tags removed but
    their inline content preserved. This is what the model actually
    "reads" at this region's positions in the base prompt.
    """
    return DETAIL_BLOCK_REGEX_PATTERN.sub(lambda match: match.group(1), region_body_text_with_detail_tags)


def _extract_whitespace_split_target_words_from_all_detail_blocks_in_one_region_body(
    region_body_text_with_detail_tags
):
    """
    Returns a flat list of whitespace-split words from every
    <DETAIL>...</DETAIL> body inside this region. Each word will be
    independently masked (mirrors BNK's target_text.split(" ")
    behavior).
    """
    accumulated_target_words_list = []
    for detail_match_object in DETAIL_BLOCK_REGEX_PATTERN.finditer(region_body_text_with_detail_tags):
        detail_inline_content_text = detail_match_object.group(1)
        for one_whitespace_split_word in detail_inline_content_text.split():
            if one_whitespace_split_word:
                accumulated_target_words_list.append(one_whitespace_split_word)
    return accumulated_target_words_list


def _normalize_runs_of_whitespace_to_single_space_and_strip_outer(raw_text_value_to_normalize):
    return re.sub(r"\s+", " ", raw_text_value_to_normalize or "").strip()


def _parse_inline_tagged_prompt_into_section_descriptors_in_textual_order(
    raw_inline_tagged_prompt_text,
):
    """
    Walks the input text in left-to-right order, splitting it into a
    sequence of section descriptors:

      - Any text BETWEEN <REGION> blocks (or outside all of them)
        becomes a passthrough section descriptor (no region, no
        targets). Empty/whitespace-only fragments are dropped.
      - Each <REGION>...</REGION> block becomes a true-region section
        descriptor with:
          enhanced_text = region body with DETAIL tags inlined
          global_text   = enhanced_text (set as the same text; the
                          encoder's set-diff path is bypassed via
                          override_target_words_list anyway, but a
                          non-empty global_text is required for the
                          "is_true_region" gate to engage)
          override_target_words_list = whitespace-split words from
                                       every DETAIL body inside this
                                       region
          is_true_region = True

    Returns a list of section descriptor dicts in declaration order
    (matching the encoder's expected shape).
    """
    parsed_section_descriptors_in_declaration_order_list = []
    current_cursor_position_in_input_text = 0
    one_based_section_index_counter_for_diagnostics = 0
    for region_block_match_object in REGION_BLOCK_WITH_OPTIONAL_ATTRS_REGEX_PATTERN.finditer(
        raw_inline_tagged_prompt_text
    ):
        # 1. Capture any passthrough text BEFORE this region block.
        passthrough_text_between_previous_region_and_this_one = raw_inline_tagged_prompt_text[
            current_cursor_position_in_input_text : region_block_match_object.start()
        ]
        passthrough_text_normalized = _normalize_runs_of_whitespace_to_single_space_and_strip_outer(
            passthrough_text_between_previous_region_and_this_one
        )
        # Treat surrounding punctuation-only (commas, semicolons, dots) as
        # noise — drop a passthrough that's just separator punctuation.
        passthrough_stripped_of_separator_punctuation = re.sub(
            r"^[\s,;.]+|[\s,;.]+$", "", passthrough_text_normalized
        )
        if passthrough_stripped_of_separator_punctuation:
            one_based_section_index_counter_for_diagnostics += 1
            parsed_section_descriptors_in_declaration_order_list.append({
                "section_id_one_based": one_based_section_index_counter_for_diagnostics,
                "global_text": passthrough_stripped_of_separator_punctuation,
                "enhanced_text": passthrough_stripped_of_separator_punctuation,
                "global_text_weight": 1.0,
                "enhanced_text_weight": 1.0,
                "clip_l_strength": 1.0,
                "clip_g_strength": 1.0,
                "is_true_region": False,
                "start_timestep": DEFAULT_TIMESTEP_RANGE_START_VALUE,
                "end_timestep": DEFAULT_TIMESTEP_RANGE_END_VALUE,
            })

        # 2. Now the region itself. Extract optional <REGION ...> attrs.
        attrs_block_text_inside_opening_tag = region_block_match_object.group(
            "attrs_block_inside_opening_tag"
        ) or ""
        parsed_region_attribute_dict = (
            _parse_optional_region_attributes_block_into_timestep_range_dict(
                attrs_block_text_inside_opening_tag
            )
        )
        this_region_start_timestep_value = _parse_one_timestep_attribute_value_to_clamped_float(
            parsed_region_attribute_dict.get("start_timestep"),
            DEFAULT_TIMESTEP_RANGE_START_VALUE,
        )
        this_region_end_timestep_value = _parse_one_timestep_attribute_value_to_clamped_float(
            parsed_region_attribute_dict.get("end_timestep"),
            DEFAULT_TIMESTEP_RANGE_END_VALUE,
        )
        # Defensive: ensure start <= end. If user typed them backwards,
        # swap silently.
        if this_region_start_timestep_value > this_region_end_timestep_value:
            (
                this_region_start_timestep_value,
                this_region_end_timestep_value,
            ) = (
                this_region_end_timestep_value,
                this_region_start_timestep_value,
            )

        region_body_with_detail_tags_raw = region_block_match_object.group(
            "region_body_between_tags"
        )
        region_visible_text_with_detail_tags_inlined = (
            _strip_detail_tags_keeping_inline_content_from_one_region_body(
                region_body_with_detail_tags_raw
            )
        )
        region_visible_text_normalized = _normalize_runs_of_whitespace_to_single_space_and_strip_outer(
            region_visible_text_with_detail_tags_inlined
        )
        if region_visible_text_normalized:
            target_words_list_for_this_region = (
                _extract_whitespace_split_target_words_from_all_detail_blocks_in_one_region_body(
                    region_body_with_detail_tags_raw
                )
            )
            one_based_section_index_counter_for_diagnostics += 1
            parsed_section_descriptors_in_declaration_order_list.append({
                "section_id_one_based": one_based_section_index_counter_for_diagnostics,
                "global_text": region_visible_text_normalized,
                "enhanced_text": region_visible_text_normalized,
                "global_text_weight": 1.0,
                "enhanced_text_weight": 1.0,
                "clip_l_strength": 1.0,
                "clip_g_strength": 1.0,
                "is_true_region": True,
                "override_target_words_list": target_words_list_for_this_region,
                "start_timestep": this_region_start_timestep_value,
                "end_timestep": this_region_end_timestep_value,
            })

        current_cursor_position_in_input_text = region_block_match_object.end()

    # 3. Any trailing passthrough text after the last region block.
    trailing_passthrough_text = raw_inline_tagged_prompt_text[
        current_cursor_position_in_input_text:
    ]
    trailing_passthrough_normalized = _normalize_runs_of_whitespace_to_single_space_and_strip_outer(
        trailing_passthrough_text
    )
    trailing_passthrough_stripped = re.sub(
        r"^[\s,;.]+|[\s,;.]+$", "", trailing_passthrough_normalized
    )
    if trailing_passthrough_stripped:
        one_based_section_index_counter_for_diagnostics += 1
        parsed_section_descriptors_in_declaration_order_list.append({
            "section_id_one_based": one_based_section_index_counter_for_diagnostics,
            "global_text": trailing_passthrough_stripped,
            "enhanced_text": trailing_passthrough_stripped,
            "global_text_weight": 1.0,
            "enhanced_text_weight": 1.0,
            "clip_l_strength": 1.0,
            "clip_g_strength": 1.0,
            "is_true_region": False,
            "start_timestep": DEFAULT_TIMESTEP_RANGE_START_VALUE,
            "end_timestep": DEFAULT_TIMESTEP_RANGE_END_VALUE,
        })

    return parsed_section_descriptors_in_declaration_order_list


INLINE_TAGGED_PROMPT_PLACEHOLDER_HELP_TEXT = (
    # Header instructions are drawn ABOVE this field by the frontend
    # extension (web/clip_text_encode_sdxl_enhanced_inline_tagged.js).
    # The placeholder stays minimal — just the example — since the
    # syntax docs are visible up there permanently.
    "Example: <REGION>a <DETAIL>gnarled old</DETAIL> tree</REGION>, "
    "<REGION>a bird <DETAIL>red eyes</DETAIL></REGION>"
)


class CLIPTextEncodeSDXLEnhancedInlineTagged:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "inline_tagged_prompt_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": INLINE_TAGGED_PROMPT_PLACEHOLDER_HELP_TEXT,
                }),
                "upscaled_conditioning_multiplier": ("FLOAT", {
                    "default": UPSCALED_CONDITIONING_MULTIPLIER_DEFAULT_VALUE,
                    "min": UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE,
                    "step": 0.01,
                }),
                "enable_region_detail_processing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "ON: parse REGION/DETAIL tags and apply cutoff isolation math. "
                        "OFF: strip tags from the prompt and encode the plain text as a "
                        "single passthrough (no isolation). Useful for A/B comparing the "
                        "isolation effect against plain CLIP encoding of the same prompt."
                    ),
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
                "join_separator": ("STRING", {
                    "default": ",",
                    "multiline": False,
                    "tooltip": (
                        "String inserted between parsed sections in the base prompt "
                        "when the prior section did not already end with this separator. "
                        "Default ','. Empty = no separator inserted; single commas still "
                        "preserved from any trailing punctuation the user typed in the section."
                    ),
                }),
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
            },
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "upscaled_conditioning", "reference_full_prompt")
    FUNCTION = "encode_inline_tagged_prompt_into_sdxl_conditioning_pair_and_reference_text"
    CATEGORY = "unified-conditioning-merge"

    def encode_inline_tagged_prompt_into_sdxl_conditioning_pair_and_reference_text(
        self,
        clip,
        inline_tagged_prompt_text,
        upscaled_conditioning_multiplier,
        enable_region_detail_processing,
        support_a1111_style_embedding_text,
        remove_text_for_unsupported_embeddings,
        filter_known_a1111_embedding_tags_not_installed_locally,
        join_separator,
        zoom,
        offset_x,
        offset_y,
        latent=None,
    ):
        raw_user_text_value = str(inline_tagged_prompt_text or "")
        if not bool(enable_region_detail_processing):
            # A/B-test mode: strip ALL <REGION>...</REGION> and <DETAIL>...</DETAIL>
            # tags from the user's text and treat the result as a single
            # passthrough section. No cutoff math runs; this should produce
            # the same conditioning as feeding the plain text into a stock
            # CLIPTextEncode of equivalent SDXL geometry.
            text_with_detail_tags_inlined = DETAIL_BLOCK_REGEX_PATTERN.sub(
                lambda match: match.group(1), raw_user_text_value
            )
            # REGION_BLOCK_WITH_OPTIONAL_ATTRS_REGEX_PATTERN's body group is
            # named "region_body_between_tags"; inline that group's content.
            text_with_region_tags_inlined_too = REGION_BLOCK_WITH_OPTIONAL_ATTRS_REGEX_PATTERN.sub(
                lambda match: match.group("region_body_between_tags"),
                text_with_detail_tags_inlined,
            )
            plain_text_with_all_tags_stripped = (
                _normalize_runs_of_whitespace_to_single_space_and_strip_outer(
                    text_with_region_tags_inlined_too
                )
            )
            parsed_section_descriptors_list = []
            if plain_text_with_all_tags_stripped:
                parsed_section_descriptors_list.append({
                    "section_id_one_based": 1,
                    "global_text": plain_text_with_all_tags_stripped,
                    "enhanced_text": plain_text_with_all_tags_stripped,
                    "global_text_weight": 1.0,
                    "enhanced_text_weight": 1.0,
                    "clip_l_strength": 1.0,
                    "clip_g_strength": 1.0,
                    "is_true_region": False,
                    "start_timestep": DEFAULT_TIMESTEP_RANGE_START_VALUE,
                    "end_timestep": DEFAULT_TIMESTEP_RANGE_END_VALUE,
                })
        else:
            parsed_section_descriptors_list = (
                _parse_inline_tagged_prompt_into_section_descriptors_in_textual_order(
                    raw_user_text_value
                )
            )

        # Apply v3-style per-section text transforms (A1111 rewrite,
        # unsupported-embedding strip, orphan-tag filter, shape-mismatch
        # warning logging) to BOTH global_text and enhanced_text of every
        # parsed section. Mutates in place.
        for section_descriptor_to_transform_text_of in parsed_section_descriptors_list:
            section_descriptor_to_transform_text_of["global_text"] = (
                apply_v3_per_text_transforms_to_one_text_string(
                    section_descriptor_to_transform_text_of.get("global_text", ""),
                    clip,
                    bool(support_a1111_style_embedding_text),
                    bool(remove_text_for_unsupported_embeddings),
                    bool(filter_known_a1111_embedding_tags_not_installed_locally),
                )
            )
            section_descriptor_to_transform_text_of["enhanced_text"] = (
                apply_v3_per_text_transforms_to_one_text_string(
                    section_descriptor_to_transform_text_of.get("enhanced_text", ""),
                    clip,
                    bool(support_a1111_style_embedding_text),
                    bool(remove_text_for_unsupported_embeddings),
                    bool(filter_known_a1111_embedding_tags_not_installed_locally),
                )
            )
        # Drop sections that ended up fully empty after transforms.
        parsed_section_descriptors_list = [
            section_descriptor for section_descriptor in parsed_section_descriptors_list
            if section_descriptor.get("global_text") or section_descriptor.get("enhanced_text")
        ]

        # SDXL geometry resolution (primary + upscaled).
        primary_target_image_width, primary_target_image_height = (
            resolve_target_image_width_and_height_from_optional_latent_or_defaults(latent)
        )
        conditioning_upscale_factor_clamped = max(
            UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE,
            float(upscaled_conditioning_multiplier),
        )
        upscaled_target_image_width = int(
            round(primary_target_image_width * conditioning_upscale_factor_clamped)
        )
        upscaled_target_image_height = int(
            round(primary_target_image_height * conditioning_upscale_factor_clamped)
        )
        zoom_factor_clamped = max(ZOOM_MINIMUM_VALUE, float(zoom))
        offset_x_clamped = clamp_numeric_value_inclusive(
            float(offset_x), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE
        )
        offset_y_clamped = clamp_numeric_value_inclusive(
            float(offset_y), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE
        )
        primary_sdxl_size_and_crop_metadata_fields = compute_sdxl_size_and_crop_metadata_fields(
            primary_target_image_width, primary_target_image_height,
            zoom_factor_clamped, offset_x_clamped, offset_y_clamped,
        )
        if (
            upscaled_target_image_width != primary_target_image_width
            or upscaled_target_image_height != primary_target_image_height
        ):
            upscaled_sdxl_size_and_crop_metadata_fields = compute_sdxl_size_and_crop_metadata_fields(
                upscaled_target_image_width, upscaled_target_image_height,
                zoom_factor_clamped, offset_x_clamped, offset_y_clamped,
            )
        else:
            upscaled_sdxl_size_and_crop_metadata_fields = primary_sdxl_size_and_crop_metadata_fields

        # Empty fallback: no sections at all.
        if not parsed_section_descriptors_list:
            empty_tokens_dict = clip.tokenize("")
            primary_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=primary_sdxl_size_and_crop_metadata_fields,
            )
            upscaled_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=upscaled_sdxl_size_and_crop_metadata_fields,
            )
            return (primary_empty_conditioning_list, upscaled_empty_conditioning_list, "")

        join_separator_string_for_this_call = str(join_separator if join_separator is not None else ",")

        # Timestep segmentation: compute unique breakpoints from the
        # union of every section's start_timestep + end_timestep. For
        # each sub-interval between adjacent breakpoints, encode using
        # ONLY the sections whose [start_timestep, end_timestep] covers
        # that sub-interval. Each segment becomes its own CONDITIONING
        # entry stamped with start_percent / end_percent — ComfyUI's
        # sampler then routes each entry only to the diffusion steps
        # whose sigma-percent falls in that range. Concat-style merge
        # semantics (see conditioning_merge_with_timestep_ranges.py).
        sorted_unique_timestep_breakpoints = sorted({
            DEFAULT_TIMESTEP_RANGE_START_VALUE,
            DEFAULT_TIMESTEP_RANGE_END_VALUE,
        } | {
            float(section_descriptor.get("start_timestep", DEFAULT_TIMESTEP_RANGE_START_VALUE))
            for section_descriptor in parsed_section_descriptors_list
        } | {
            float(section_descriptor.get("end_timestep", DEFAULT_TIMESTEP_RANGE_END_VALUE))
            for section_descriptor in parsed_section_descriptors_list
        })

        primary_output_conditioning_entries_list = []
        upscaled_output_conditioning_entries_list = []
        reference_text_lines_per_sub_interval_for_display = []

        for breakpoint_zero_based_index_into_sorted_list in range(
            len(sorted_unique_timestep_breakpoints) - 1
        ):
            sub_interval_start_timestep_value = sorted_unique_timestep_breakpoints[
                breakpoint_zero_based_index_into_sorted_list
            ]
            sub_interval_end_timestep_value = sorted_unique_timestep_breakpoints[
                breakpoint_zero_based_index_into_sorted_list + 1
            ]
            if (sub_interval_end_timestep_value - sub_interval_start_timestep_value
                    <= FLOAT_EPSILON_FOR_TIMESTEP_BREAKPOINT_DEDUPE):
                continue

            # Section is active in this sub-interval iff its
            # [start_timestep, end_timestep] fully covers it.
            sections_active_in_this_sub_interval_list = [
                section_descriptor for section_descriptor in parsed_section_descriptors_list
                if (
                    float(section_descriptor.get("start_timestep", DEFAULT_TIMESTEP_RANGE_START_VALUE))
                    <= sub_interval_start_timestep_value + FLOAT_EPSILON_FOR_TIMESTEP_BREAKPOINT_DEDUPE
                ) and (
                    float(section_descriptor.get("end_timestep", DEFAULT_TIMESTEP_RANGE_END_VALUE))
                    >= sub_interval_end_timestep_value - FLOAT_EPSILON_FOR_TIMESTEP_BREAKPOINT_DEDUPE
                )
            ]
            if not sections_active_in_this_sub_interval_list:
                continue

            try:
                raw_conditioning_entry_for_this_sub_interval = (
                    encode_active_v3_sections_into_one_sdxl_conditioning_entry(
                        clip,
                        sections_active_in_this_sub_interval_list,
                        join_separator_string_for_this_call,
                    )
                )
            except Exception as encoding_failure_exception_caught:
                logging.warning(
                    f"CLIPTextEncodeSDXLEnhancedInlineTagged: encoding failed for "
                    f"sub-interval [{sub_interval_start_timestep_value}, "
                    f"{sub_interval_end_timestep_value}] "
                    f"({type(encoding_failure_exception_caught).__name__}: "
                    f"{encoding_failure_exception_caught}). Skipping this sub-interval."
                )
                continue

            raw_tokens_tensor_for_this_sub_interval = (
                raw_conditioning_entry_for_this_sub_interval[0]
            )
            raw_metadata_dict_for_this_sub_interval = (
                raw_conditioning_entry_for_this_sub_interval[1]
            )

            primary_entry_metadata_dict_for_this_sub_interval = dict(
                raw_metadata_dict_for_this_sub_interval
            )
            primary_entry_metadata_dict_for_this_sub_interval.update(
                primary_sdxl_size_and_crop_metadata_fields
            )
            primary_entry_metadata_dict_for_this_sub_interval["start_percent"] = (
                float(sub_interval_start_timestep_value)
            )
            primary_entry_metadata_dict_for_this_sub_interval["end_percent"] = (
                float(sub_interval_end_timestep_value)
            )

            upscaled_entry_metadata_dict_for_this_sub_interval = dict(
                raw_metadata_dict_for_this_sub_interval
            )
            upscaled_entry_metadata_dict_for_this_sub_interval.update(
                upscaled_sdxl_size_and_crop_metadata_fields
            )
            upscaled_entry_metadata_dict_for_this_sub_interval["start_percent"] = (
                float(sub_interval_start_timestep_value)
            )
            upscaled_entry_metadata_dict_for_this_sub_interval["end_percent"] = (
                float(sub_interval_end_timestep_value)
            )

            primary_output_conditioning_entries_list.append(
                [
                    raw_tokens_tensor_for_this_sub_interval,
                    primary_entry_metadata_dict_for_this_sub_interval,
                ]
            )
            upscaled_output_conditioning_entries_list.append(
                [
                    raw_tokens_tensor_for_this_sub_interval,
                    upscaled_entry_metadata_dict_for_this_sub_interval,
                ]
            )

            per_sub_interval_reference_text_value = (
                build_plain_text_reference_prompt_without_clip_weight_wrapping_for_display(
                    sections_active_in_this_sub_interval_list,
                    join_separator_string_for_this_call,
                )
            )
            reference_text_lines_per_sub_interval_for_display.append(
                f"[{sub_interval_start_timestep_value:.3f}..{sub_interval_end_timestep_value:.3f}] "
                f"{per_sub_interval_reference_text_value}"
            )

        # All sub-intervals failed or were empty: fall back to empty.
        if not primary_output_conditioning_entries_list:
            empty_tokens_dict = clip.tokenize("")
            primary_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=primary_sdxl_size_and_crop_metadata_fields,
            )
            upscaled_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=upscaled_sdxl_size_and_crop_metadata_fields,
            )
            return (primary_empty_conditioning_list, upscaled_empty_conditioning_list, "")

        reference_plain_text_for_user_display = "\n".join(
            reference_text_lines_per_sub_interval_for_display
        )

        return (
            primary_output_conditioning_entries_list,
            upscaled_output_conditioning_entries_list,
            reference_plain_text_for_user_display,
        )
