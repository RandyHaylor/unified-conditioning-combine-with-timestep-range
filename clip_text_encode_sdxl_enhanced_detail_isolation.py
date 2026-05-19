"""
CLIP Text Encode SDXL Enhanced (Detail Isolation) — primary node.

Consumes a chain of CLIPTextEncodeSDXLEnhancedDetailIsolationSection
nodes (typed `DETAIL_ISOLATION_SECTION_CHAIN`) and produces SDXL
conditioning using the same v3 cutoff math as
CLIPTextEncodeSDXLV3GlobalAndEnhanced — just with sections supplied
through node-graph chaining instead of widgets.

The primary node carries ONLY global / image-geometry settings:
  - upscaled_conditioning_multiplier
  - zoom, offset_x, offset_y (zoom-effect group)
  - optional latent input

Per-section settings (global_text, enhanced_text, weights, L/G strengths,
A1111 toggles) all live on the section nodes. Section validator widgets
also live on the section nodes (separate frontend extension).

Imports v3's shared encoding helpers via the public aliases defined at
the bottom of v3's module — clean cross-module reuse, no reaching into
private-prefixed internals.
"""

import logging

import nodes

from .clip_text_encode_sdxl_enhanced_detail_isolation_section import (
    DETAIL_ISOLATION_SECTION_CHAIN_TYPE_NAME,
)
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


def _apply_per_section_text_transforms_in_place_using_each_section_descriptors_own_toggle_values(
    section_descriptor_list_from_chain, clip_object
):
    """
    Each section descriptor in the chain carries its OWN per-section
    A1111-related toggle values (different sections can use different
    settings). Apply each section's transforms to its own global_text +
    enhanced_text using THAT section's toggle values. Mutates in place.
    After transforms, re-evaluate `is_true_region` based on the
    post-transform text values.
    """
    for one_section_descriptor in section_descriptor_list_from_chain:
        per_section_support_a1111_setting = bool(
            one_section_descriptor.get("support_a1111_style_embedding_text", True)
        )
        per_section_remove_unsupported_setting = bool(
            one_section_descriptor.get("remove_text_for_unsupported_embeddings", True)
        )
        per_section_filter_orphan_setting = bool(
            one_section_descriptor.get("filter_known_a1111_embedding_tags_not_installed_locally", True)
        )
        one_section_descriptor["global_text"] = (
            apply_v3_per_text_transforms_to_one_text_string(
                one_section_descriptor.get("global_text", ""),
                clip_object,
                per_section_support_a1111_setting,
                per_section_remove_unsupported_setting,
                per_section_filter_orphan_setting,
            )
        )
        one_section_descriptor["enhanced_text"] = (
            apply_v3_per_text_transforms_to_one_text_string(
                one_section_descriptor.get("enhanced_text", ""),
                clip_object,
                per_section_support_a1111_setting,
                per_section_remove_unsupported_setting,
                per_section_filter_orphan_setting,
            )
        )
        post_transform_global_text_lower = (one_section_descriptor.get("global_text") or "").lower()
        post_transform_enhanced_text_lower = (one_section_descriptor.get("enhanced_text") or "").lower()
        one_section_descriptor["is_true_region"] = (
            bool(post_transform_enhanced_text_lower)
            and post_transform_enhanced_text_lower != post_transform_global_text_lower
        )


def _filter_out_sections_whose_text_is_now_fully_empty_after_transforms(section_descriptor_list):
    return [
        section_descriptor for section_descriptor in section_descriptor_list
        if section_descriptor.get("global_text") or section_descriptor.get("enhanced_text")
    ]


class CLIPTextEncodeSDXLEnhancedDetailIsolation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "upscaled_conditioning_multiplier": ("FLOAT", {
                    "default": UPSCALED_CONDITIONING_MULTIPLIER_DEFAULT_VALUE,
                    "min": UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE,
                    "step": 0.01,
                }),
                "join_separator": ("STRING", {
                    "default": ",",
                    "multiline": False,
                    "tooltip": (
                        "String inserted between chained sections in the base prompt "
                        "when the prior section did not already end with this separator. "
                        "Default ','. Empty = no separator inserted (sections concatenate "
                        "directly; single commas still preserved from typed trailing punctuation)."
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
                "prompt_sections": (DETAIL_ISOLATION_SECTION_CHAIN_TYPE_NAME,),
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "upscaled_conditioning", "reference_full_prompt")
    FUNCTION = "encode_chained_sections_into_sdxl_conditioning_pair_and_reference_prompt"
    CATEGORY = "unified-conditioning-merge"

    def encode_chained_sections_into_sdxl_conditioning_pair_and_reference_prompt(
        self,
        clip,
        upscaled_conditioning_multiplier,
        join_separator,
        zoom,
        offset_x,
        offset_y,
        prompt_sections=None,
        latent=None,
    ):
        section_descriptors_from_chain_list = (
            list(prompt_sections) if prompt_sections is not None else []
        )
        # Drop sections where BOTH texts are empty before doing any work.
        section_descriptors_from_chain_list = [
            section_descriptor for section_descriptor in section_descriptors_from_chain_list
            if (section_descriptor.get("global_text") or "").strip()
            or (section_descriptor.get("enhanced_text") or "").strip()
        ]
        # Per-section text transforms — uses each section's own toggles.
        _apply_per_section_text_transforms_in_place_using_each_section_descriptors_own_toggle_values(
            section_descriptors_from_chain_list, clip
        )
        section_descriptors_from_chain_list = (
            _filter_out_sections_whose_text_is_now_fully_empty_after_transforms(
                section_descriptors_from_chain_list
            )
        )

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

        # Empty-prompt fallback: no active sections at all.
        if not section_descriptors_from_chain_list:
            empty_tokens_dict = clip.tokenize("")
            primary_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=primary_sdxl_size_and_crop_metadata_fields,
            )
            upscaled_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=upscaled_sdxl_size_and_crop_metadata_fields,
            )
            return (primary_empty_conditioning_list, upscaled_empty_conditioning_list, "")

        join_separator_string_for_this_call = str(join_separator if join_separator is not None else ",")
        # v3 cutoff math via the shared encoder.
        try:
            raw_conditioning_entry_from_shared_encoder = (
                encode_active_v3_sections_into_one_sdxl_conditioning_entry(
                    clip, section_descriptors_from_chain_list, join_separator_string_for_this_call
                )
            )
        except Exception as encoding_failure_exception_caught:
            logging.warning(
                f"CLIPTextEncodeSDXLEnhancedDetailIsolation: encoding failed "
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
                section_descriptors_from_chain_list, join_separator_string_for_this_call
            )
        )

        return (
            [[raw_tokens_tensor, primary_entry_metadata_dict]],
            [[raw_tokens_tensor, upscaled_entry_metadata_dict]],
            reference_plain_text_for_user_display,
        )
