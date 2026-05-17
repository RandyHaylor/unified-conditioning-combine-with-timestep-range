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
# body can span newlines).
REGION_BLOCK_REGEX_PATTERN = re.compile(
    r"<\s*REGION\s*>(.*?)<\s*/\s*REGION\s*>", re.IGNORECASE | re.DOTALL
)
# Capture <DETAIL>...</DETAIL> blocks inside a region body.
DETAIL_BLOCK_REGEX_PATTERN = re.compile(
    r"<\s*DETAIL\s*>(.*?)<\s*/\s*DETAIL\s*>", re.IGNORECASE | re.DOTALL
)


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
    for region_block_match_object in REGION_BLOCK_REGEX_PATTERN.finditer(raw_inline_tagged_prompt_text):
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
            })

        # 2. Now the region itself.
        region_body_with_detail_tags_raw = region_block_match_object.group(1)
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
                # Both fields hold the SAME visible region text; the encoder
                # treats this as a true region because is_true_region is
                # forced True below. The target words bypass set-diff via
                # override_target_words_list.
                "global_text": region_visible_text_normalized,
                "enhanced_text": region_visible_text_normalized,
                "global_text_weight": 1.0,
                "enhanced_text_weight": 1.0,
                "clip_l_strength": 1.0,
                "clip_g_strength": 1.0,
                "is_true_region": True,
                "override_target_words_list": target_words_list_for_this_region,
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
                "support_a1111_style_embedding_text": ("BOOLEAN", {"default": True}),
                "remove_text_for_unsupported_embeddings": ("BOOLEAN", {"default": True}),
                "filter_known_a1111_embedding_tags_not_installed_locally": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "List can be modified in custom node folder: "
                        "known_a1111_embedding_names_to_filter_when_not_installed_locally.txt"
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
        support_a1111_style_embedding_text,
        remove_text_for_unsupported_embeddings,
        filter_known_a1111_embedding_tags_not_installed_locally,
        zoom,
        offset_x,
        offset_y,
        latent=None,
    ):
        parsed_section_descriptors_list = (
            _parse_inline_tagged_prompt_into_section_descriptors_in_textual_order(
                str(inline_tagged_prompt_text or "")
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

        try:
            raw_conditioning_entry_from_shared_encoder = (
                encode_active_v3_sections_into_one_sdxl_conditioning_entry(
                    clip, parsed_section_descriptors_list
                )
            )
        except Exception as encoding_failure_exception_caught:
            logging.warning(
                f"CLIPTextEncodeSDXLEnhancedInlineTagged: encoding failed "
                f"({type(encoding_failure_exception_caught).__name__}: "
                f"{encoding_failure_exception_caught}). Falling back to empty conditioning."
            )
            empty_tokens_dict = clip.tokenize("")
            primary_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=primary_sdxl_size_and_crop_metadata_fields,
            )
            upscaled_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=upscaled_sdxl_size_and_crop_metadata_fields,
            )
            return (primary_empty_conditioning_list, upscaled_empty_conditioning_list, "")

        raw_tokens_tensor = raw_conditioning_entry_from_shared_encoder[0]
        raw_metadata_dict = raw_conditioning_entry_from_shared_encoder[1]
        primary_entry_metadata_dict = dict(raw_metadata_dict)
        primary_entry_metadata_dict.update(primary_sdxl_size_and_crop_metadata_fields)
        upscaled_entry_metadata_dict = dict(raw_metadata_dict)
        upscaled_entry_metadata_dict.update(upscaled_sdxl_size_and_crop_metadata_fields)

        reference_plain_text_for_user_display = (
            build_plain_text_reference_prompt_without_clip_weight_wrapping_for_display(
                parsed_section_descriptors_list
            )
        )

        return (
            [[raw_tokens_tensor, primary_entry_metadata_dict]],
            [[raw_tokens_tensor, upscaled_entry_metadata_dict]],
            reference_plain_text_for_user_display,
        )
