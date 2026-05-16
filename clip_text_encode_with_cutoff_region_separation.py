"""
CLIP Text Encode (Cutoff Region Separation)

Single-node prompt builder using a self-contained, MIT-licensed Cutoff-style
algorithm (see cutoff_per_stream_isolation.py) that runs INDEPENDENTLY per
CLIP stream. This is what lets per-section CLIP routing actually work
correctly for SDXL: when a section is marked "L only", the L stream encodes
that section's text and the G stream encodes the empty prompt (giving
natural CLIP-G empty embeddings, NOT zero-masked tensors that would be
out-of-distribution for the model).

For each section, the dropdown chooses which stream(s) this section
contributes to:
    L+G   — section text goes to BOTH text_l and text_g (typical)
    L     — section text goes to text_l only; text_g for this section's group
            is the empty string (CLIP-G encodes the natural empty prompt)
    G     — symmetric for G

Sections sharing a clip choice are GROUPED. Per group, we build the
per-stream prompts and run cutoff-style isolation independently on each
stream. The L-stream and G-stream results are concatenated along the last
dim to form the SDXL combined tensor (768 + 1280 = 2048).

All groups' resulting CONDITIONING entries are emitted together as a
multi-entry CONDITIONING (combine-style — sampler treats each group as a
parallel branch).

SDXL size/crop metadata (zoom + offsets, target W/H from optional LATENT
defaulting to 1024x1024) is stamped onto every output entry.

NO external plugin dependency. Self-contained.
"""

import inspect
import logging
import re
import sys

import torch

from .cutoff_per_stream_isolation import encode_one_stream_with_cutoff_style_region_isolation


MAX_SECTION_COUNT_SUPPORTED = 16
DEFAULT_SECTION_COUNT_VALUE = 3

CLIP_STREAM_PASS_BOTH_L_AND_G = "Pass L+G"
CLIP_STREAM_PASS_L_ONLY = "Pass L"
CLIP_STREAM_PASS_G_ONLY = "Pass G"
CLIP_STREAM_PASS_CLASSIC_UPSTREAM_CUTOFF = "Classic"
CLIP_STREAM_PASS_CHOICES_IN_DROPDOWN_ORDER = [
    CLIP_STREAM_PASS_BOTH_L_AND_G,
    CLIP_STREAM_PASS_L_ONLY,
    CLIP_STREAM_PASS_G_ONLY,
    CLIP_STREAM_PASS_CLASSIC_UPSTREAM_CUTOFF,
]

LATENT_TO_IMAGE_SPATIAL_SCALE_FACTOR = 8
DEFAULT_LATENT_IMAGE_WIDTH_WHEN_NO_LATENT_INPUT = 1024
DEFAULT_LATENT_IMAGE_HEIGHT_WHEN_NO_LATENT_INPUT = 1024

ZOOM_MINIMUM_VALUE = 1.0
ZOOM_MAXIMUM_VALUE = 100.0
ZOOM_DEFAULT_VALUE = 1.0
OFFSET_MINIMUM_VALUE = -1.0
OFFSET_MAXIMUM_VALUE = 1.0
OFFSET_DEFAULT_VALUE = 0.0

WHITESPACE_RUN_REGEX_FOR_NORMALIZING_SECTION_TEXT = re.compile(r"\s+")


# -------------- Classic mode: upstream ComfyUI_Cutoff plugin lookup --------------

def _find_loaded_upstream_cutoff_module_in_sys_modules_or_none():
    """
    Locate the ComfyUI_Cutoff plugin's `cutoff` module by scanning sys.modules
    for one that exposes the expected interface AS A REAL Python class +
    callable function (to avoid matching torch._OpNamespace and friends whose
    __getattr__ returns truthy for any name). Prefer modules whose name
    contains "cutoff" (case-insensitive). Used ONLY when a section is marked
    with the Classic clip_pass_choice — other modes use our self-contained
    per-stream implementation.
    """
    matching_candidate_modules_keyed_by_name = {}
    for module_name_in_sys_modules, loaded_module_object in list(sys.modules.items()):
        if loaded_module_object is None:
            continue
        try:
            finalize_attribute_or_none = getattr(loaded_module_object, "finalize_clip_regions", None)
            base_prompt_attribute_or_none = getattr(loaded_module_object, "CLIPRegionsBasePrompt", None)
        except Exception:
            continue
        if finalize_attribute_or_none is None or base_prompt_attribute_or_none is None:
            continue
        if not callable(finalize_attribute_or_none) or not inspect.isclass(base_prompt_attribute_or_none):
            continue
        matching_candidate_modules_keyed_by_name[module_name_in_sys_modules] = loaded_module_object
    if not matching_candidate_modules_keyed_by_name:
        return None
    for candidate_module_name, candidate_module in matching_candidate_modules_keyed_by_name.items():
        if "cutoff" in candidate_module_name.lower():
            return candidate_module
    return next(iter(matching_candidate_modules_keyed_by_name.values()))


def _encode_one_group_via_classic_upstream_cutoff_plugin(
    clip_object,
    section_descriptors_in_classic_group_list,
    join_separator_string,
    mask_token_string,
    strict_mask_value,
    start_from_masked_value,
):
    """
    Runs the upstream ComfyUI_Cutoff plugin directly for a group of sections
    marked Classic. Mathematically identical to using the upstream plugin
    standalone — for A/B comparison against our per-stream implementation.

    Raises RuntimeError if the upstream plugin isn't installed.
    """
    upstream_cutoff_module = _find_loaded_upstream_cutoff_module_in_sys_modules_or_none()
    if upstream_cutoff_module is None:
        raise RuntimeError(
            "CLIPTextEncodeWithCutoffRegionSeparation: Classic clip_pass_choice "
            "requires the ComfyUI_Cutoff plugin to be installed (it's used as "
            "the reference implementation for A/B comparison). Install from "
            "https://github.com/BlenderNeko/ComfyUI_Cutoff into ComfyUI/custom_nodes/ "
            "and restart ComfyUI. Other clip_pass_choice options (Pass L+G / Pass L / "
            "Pass G) don't need it — they use our self-contained per-stream code."
        )

    classic_group_full_prompt_text = _build_full_prompt_text_for_one_group_of_sections(
        section_descriptors_in_classic_group_list, join_separator_string
    )

    base_prompt_node_instance = upstream_cutoff_module.CLIPRegionsBasePrompt()
    base_state_tuple_from_init_prompt = base_prompt_node_instance.init_prompt(
        clip_object, classic_group_full_prompt_text
    )
    current_clip_regions_state = base_state_tuple_from_init_prompt[0]

    add_region_node_instance = upstream_cutoff_module.CLIPSetRegion()
    for section_descriptor_in_classic_group in section_descriptors_in_classic_group_list:
        if not section_descriptor_in_classic_group["isolate"]:
            continue
        try:
            next_state_tuple_after_region_add = add_region_node_instance.add_clip_region(
                current_clip_regions_state,
                section_descriptor_in_classic_group["text"],
                section_descriptor_in_classic_group["text"],
                section_descriptor_in_classic_group["weight"],
            )
            current_clip_regions_state = next_state_tuple_after_region_add[0]
        except Exception as upstream_cutoff_register_region_exception:
            logging.warning(
                "CLIPTextEncodeWithCutoffRegionSeparation (Classic): skipped isolate "
                f"registration for section text {section_descriptor_in_classic_group['text']!r} "
                f"(upstream Cutoff raised {type(upstream_cutoff_register_region_exception).__name__}: "
                f"{upstream_cutoff_register_region_exception})."
            )

    finalize_return_tuple_from_upstream_cutoff = upstream_cutoff_module.finalize_clip_regions(
        current_clip_regions_state,
        mask_token_string,
        float(strict_mask_value),
        float(start_from_masked_value),
    )
    # Upstream returns a CONDITIONING list. For Classic we just take that list
    # as-is and let the caller stamp SDXL size/crop metadata onto each entry.
    return finalize_return_tuple_from_upstream_cutoff[0]


# -------------- section collection --------------

def _collect_active_non_empty_sections_from_kwargs(kwargs_dict, active_section_count):
    active_section_descriptors_list = []
    for section_index in range(1, int(active_section_count) + 1):
        section_text_raw = kwargs_dict.get(f"section_{section_index}_text", "")
        section_isolate_raw = kwargs_dict.get(f"section_{section_index}_isolate", True)
        section_weight_raw = kwargs_dict.get(f"section_{section_index}_weight", 1.0)
        section_clip_pass_choice_raw = kwargs_dict.get(
            f"section_{section_index}_clip", CLIP_STREAM_PASS_BOTH_L_AND_G
        )
        section_text_normalized_whitespace = WHITESPACE_RUN_REGEX_FOR_NORMALIZING_SECTION_TEXT.sub(
            " ", (section_text_raw or "")
        ).strip()
        if not section_text_normalized_whitespace:
            continue
        active_section_descriptors_list.append({
            "text": section_text_normalized_whitespace,
            "isolate": bool(section_isolate_raw),
            "weight": float(section_weight_raw),
            "clip_pass_choice": str(section_clip_pass_choice_raw),
        })
    return active_section_descriptors_list


def _build_effective_per_section_prompt_text_with_optional_clip_attention_weight_wrap(section_descriptor):
    section_bare_text = section_descriptor["text"]
    if section_descriptor["isolate"]:
        return section_bare_text
    section_weight_value = section_descriptor["weight"]
    if abs(section_weight_value - 1.0) < 1e-9:
        return section_bare_text
    return f"({section_bare_text}:{section_weight_value:.3f})"


def _build_full_prompt_text_for_one_group_of_sections(section_descriptors_in_group_list, join_separator_string):
    per_section_effective_prompt_text_fragments = [
        _build_effective_per_section_prompt_text_with_optional_clip_attention_weight_wrap(d)
        for d in section_descriptors_in_group_list
    ]
    return join_separator_string.join(per_section_effective_prompt_text_fragments)


def _group_active_sections_by_their_clip_pass_choice(active_section_descriptors_list):
    grouped_sections_by_clip_pass_choice = {}
    for section_descriptor in active_section_descriptors_list:
        clip_pass_choice_for_this_section = section_descriptor["clip_pass_choice"]
        if clip_pass_choice_for_this_section not in grouped_sections_by_clip_pass_choice:
            grouped_sections_by_clip_pass_choice[clip_pass_choice_for_this_section] = []
        grouped_sections_by_clip_pass_choice[clip_pass_choice_for_this_section].append(section_descriptor)
    return grouped_sections_by_clip_pass_choice


# -------------- per-group encoding using per-stream cutoff --------------

def _build_isolate_target_text_and_weight_pairs_for_one_group(section_descriptors_in_group_list):
    """
    Returns [{"target_text": str, "weight": float}, ...] for sections in the
    group that have isolate=True. Non-isolate sections are skipped (their
    weight is applied via the inline `(text:weight)` wrap in the group's
    joined prompt text).
    """
    isolate_target_text_and_weight_pairs_list = []
    for section_descriptor in section_descriptors_in_group_list:
        if not section_descriptor["isolate"]:
            continue
        isolate_target_text_and_weight_pairs_list.append({
            "target_text": section_descriptor["text"],
            "weight": section_descriptor["weight"],
        })
    return isolate_target_text_and_weight_pairs_list


def _pad_per_stream_tokenization_to_match_chunk_count(
    clip_object, embedding_tensor_for_short_stream, target_chunk_count, stream_key_to_pad_with_empty
):
    """
    If a stream's tokenization produced fewer chunks than the other stream,
    SDXL's combine requires matching chunk counts. We pad with empty-prompt
    encodings of the SHORT stream. Returns the padded embedding tensor.

    For now we return the tensor unchanged and let the SDXL combine
    truncate via min() — chunk-count mismatch is a corner case and most
    prompts fit in one chunk for both streams.
    """
    return embedding_tensor_for_short_stream


def _encode_one_group_into_one_sdxl_conditioning_entry(
    clip_object,
    section_descriptors_in_group_list,
    clip_pass_choice_for_this_group,
    join_separator_string,
    mask_token_string,
    strict_mask_value,
    start_from_masked_value,
):
    """
    Returns a single CONDITIONING entry [combined_tokens_tensor, metadata_dict]
    for one group's combined prompt encoded with proper per-stream handling
    (no zero-masking — empty streams get natural CLIP empty-prompt
    encodings).
    """
    group_full_prompt_text_for_isolate_streams = _build_full_prompt_text_for_one_group_of_sections(
        section_descriptors_in_group_list, join_separator_string
    )
    isolate_target_text_and_weight_pairs_for_this_group = _build_isolate_target_text_and_weight_pairs_for_one_group(
        section_descriptors_in_group_list
    )

    # Decide each stream's prompt based on the group's CLIP pass choice.
    if clip_pass_choice_for_this_group == CLIP_STREAM_PASS_BOTH_L_AND_G:
        prompt_text_for_l_stream = group_full_prompt_text_for_isolate_streams
        prompt_text_for_g_stream = group_full_prompt_text_for_isolate_streams
        run_isolation_on_l_stream = True
        run_isolation_on_g_stream = True
    elif clip_pass_choice_for_this_group == CLIP_STREAM_PASS_L_ONLY:
        prompt_text_for_l_stream = group_full_prompt_text_for_isolate_streams
        prompt_text_for_g_stream = ""
        run_isolation_on_l_stream = True
        run_isolation_on_g_stream = False
    elif clip_pass_choice_for_this_group == CLIP_STREAM_PASS_G_ONLY:
        prompt_text_for_l_stream = ""
        prompt_text_for_g_stream = group_full_prompt_text_for_isolate_streams
        run_isolation_on_l_stream = False
        run_isolation_on_g_stream = True
    else:
        raise ValueError(
            f"Unknown clip_pass_choice {clip_pass_choice_for_this_group!r}"
        )

    # Run cutoff-style isolation per stream. For the "empty" stream we pass
    # an empty isolate list so it just encodes the empty prompt normally.
    final_embedding_tensor_for_l_stream, _l_pooled_unused = encode_one_stream_with_cutoff_style_region_isolation(
        clip_object,
        "l",
        prompt_text_for_l_stream,
        isolate_target_text_and_weight_pairs_for_this_group if run_isolation_on_l_stream else [],
        mask_token_string,
        strict_mask_value,
        start_from_masked_value,
    )
    final_embedding_tensor_for_g_stream, g_pooled_output_tensor_or_none = encode_one_stream_with_cutoff_style_region_isolation(
        clip_object,
        "g",
        prompt_text_for_g_stream,
        isolate_target_text_and_weight_pairs_for_this_group if run_isolation_on_g_stream else [],
        mask_token_string,
        strict_mask_value,
        start_from_masked_value,
    )

    # SDXL combine: concat L (768) + G (1280) along last dim. Truncate to
    # shorter sequence length to handle chunk-count mismatches.
    cut_to_shared_sequence_length = min(
        final_embedding_tensor_for_l_stream.shape[1],
        final_embedding_tensor_for_g_stream.shape[1],
    )
    sdxl_combined_tokens_tensor = torch.cat(
        [
            final_embedding_tensor_for_l_stream[:, :cut_to_shared_sequence_length],
            final_embedding_tensor_for_g_stream[:, :cut_to_shared_sequence_length],
        ],
        dim=-1,
    )

    entry_metadata_dict = {}
    if g_pooled_output_tensor_or_none is not None:
        entry_metadata_dict["pooled_output"] = g_pooled_output_tensor_or_none

    return [sdxl_combined_tokens_tensor, entry_metadata_dict]


# -------------- SDXL size/crop metadata --------------

def _clamp_numeric_value_inclusive(value_to_clamp, minimum_allowed, maximum_allowed):
    return max(minimum_allowed, min(maximum_allowed, value_to_clamp))


def _compute_sdxl_size_and_crop_metadata_fields(
    target_image_width, target_image_height, zoom_factor, offset_x_clamped, offset_y_clamped
):
    source_image_width = int(round(target_image_width * zoom_factor))
    source_image_height = int(round(target_image_height * zoom_factor))
    maximum_horizontal_crop_offset = max(0, source_image_width - target_image_width)
    maximum_vertical_crop_offset = max(0, source_image_height - target_image_height)
    crop_w_value = int(round(((offset_x_clamped + 1.0) * 0.5) * maximum_horizontal_crop_offset))
    crop_h_value = int(round(((offset_y_clamped + 1.0) * 0.5) * maximum_vertical_crop_offset))
    return {
        "width": source_image_width,
        "height": source_image_height,
        "crop_w": crop_w_value,
        "crop_h": crop_h_value,
        "target_width": target_image_width,
        "target_height": target_image_height,
        "original_size_as_tuple": (source_image_width, source_image_height),
        "crop_coords_top_left": (crop_w_value, crop_h_value),
        "target_size_as_tuple": (target_image_width, target_image_height),
    }


def _resolve_target_image_width_and_height_from_optional_latent_or_defaults(latent_or_none):
    if latent_or_none is None:
        return (DEFAULT_LATENT_IMAGE_WIDTH_WHEN_NO_LATENT_INPUT, DEFAULT_LATENT_IMAGE_HEIGHT_WHEN_NO_LATENT_INPUT)
    latent_samples_tensor = latent_or_none.get("samples")
    if latent_samples_tensor is None:
        return (DEFAULT_LATENT_IMAGE_WIDTH_WHEN_NO_LATENT_INPUT, DEFAULT_LATENT_IMAGE_HEIGHT_WHEN_NO_LATENT_INPUT)
    target_image_width = latent_samples_tensor.shape[3] * LATENT_TO_IMAGE_SPATIAL_SCALE_FACTOR
    target_image_height = latent_samples_tensor.shape[2] * LATENT_TO_IMAGE_SPATIAL_SCALE_FACTOR
    return (target_image_width, target_image_height)


# -------------- the node class --------------

class CLIPTextEncodeWithCutoffRegionSeparation:
    @classmethod
    def INPUT_TYPES(cls):
        required_inputs_dict = {
            "clip": ("CLIP",),
            "section_count": ("INT", {
                "default": DEFAULT_SECTION_COUNT_VALUE,
                "min": 1,
                "max": MAX_SECTION_COUNT_SUPPORTED,
                "step": 1,
            }),
            "join_separator": ("STRING", {"multiline": False, "default": ","}),
            "mask_token": ("STRING", {"multiline": False, "default": ""}),
            "strict_mask": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "start_from_masked": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
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
        for section_index_for_declaration in range(1, MAX_SECTION_COUNT_SUPPORTED + 1):
            required_inputs_dict[f"section_{section_index_for_declaration}_text"] = (
                "STRING", {"multiline": True, "default": ""},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_isolate"] = (
                "BOOLEAN", {"default": True},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_weight"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_clip"] = (
                CLIP_STREAM_PASS_CHOICES_IN_DROPDOWN_ORDER,
                {"default": CLIP_STREAM_PASS_BOTH_L_AND_G},
            )
        return {
            "required": required_inputs_dict,
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "reference_full_prompt")
    FUNCTION = "encode_with_grouped_per_stream_cutoff_and_sdxl_zoom"
    CATEGORY = "unified-conditioning-merge"

    def encode_with_grouped_per_stream_cutoff_and_sdxl_zoom(
        self,
        clip,
        section_count,
        join_separator,
        mask_token,
        strict_mask,
        start_from_masked,
        zoom,
        offset_x,
        offset_y,
        latent=None,
        **kwargs_for_individual_section_widget_values,
    ):
        active_section_descriptors_list = _collect_active_non_empty_sections_from_kwargs(
            kwargs_for_individual_section_widget_values, section_count
        )

        target_image_width, target_image_height = _resolve_target_image_width_and_height_from_optional_latent_or_defaults(latent)
        zoom_factor_clamped = max(ZOOM_MINIMUM_VALUE, float(zoom))
        offset_x_clamped = _clamp_numeric_value_inclusive(float(offset_x), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE)
        offset_y_clamped = _clamp_numeric_value_inclusive(float(offset_y), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE)
        sdxl_size_and_crop_metadata_fields = _compute_sdxl_size_and_crop_metadata_fields(
            target_image_width, target_image_height, zoom_factor_clamped, offset_x_clamped, offset_y_clamped
        )

        if not active_section_descriptors_list:
            # Encode empty prompt and return single empty entry.
            try:
                empty_conditioning_entry = _encode_one_group_into_one_sdxl_conditioning_entry(
                    clip, [], CLIP_STREAM_PASS_BOTH_L_AND_G, join_separator,
                    mask_token, float(strict_mask), float(start_from_masked),
                )
                empty_conditioning_entry[1].update(sdxl_size_and_crop_metadata_fields)
                return ([empty_conditioning_entry], "")
            except Exception as encoding_failure_for_empty_prompt:
                logging.warning(
                    f"CLIPTextEncodeWithCutoffRegionSeparation: empty-prompt encode failed: "
                    f"{encoding_failure_for_empty_prompt}"
                )
                return ([], "")

        grouped_sections_by_clip_pass_choice = _group_active_sections_by_their_clip_pass_choice(
            active_section_descriptors_list
        )

        output_conditioning_entries_combined_across_groups = []
        per_group_full_prompt_text_for_reference_display = []
        for clip_pass_choice_for_this_group, section_descriptors_in_this_group in grouped_sections_by_clip_pass_choice.items():
            try:
                if clip_pass_choice_for_this_group == CLIP_STREAM_PASS_CLASSIC_UPSTREAM_CUTOFF:
                    # Classic group: route through the upstream ComfyUI_Cutoff
                    # plugin directly. Mathematically identical to using the
                    # upstream plugin standalone — for A/B comparison.
                    classic_group_conditioning_list = _encode_one_group_via_classic_upstream_cutoff_plugin(
                        clip,
                        section_descriptors_in_this_group,
                        join_separator,
                        mask_token,
                        float(strict_mask),
                        float(start_from_masked),
                    )
                    for classic_group_entry in classic_group_conditioning_list:
                        classic_group_entry_tokens_tensor = classic_group_entry[0]
                        classic_group_entry_metadata_dict = dict(classic_group_entry[1])
                        classic_group_entry_metadata_dict.update(sdxl_size_and_crop_metadata_fields)
                        output_conditioning_entries_combined_across_groups.append(
                            [classic_group_entry_tokens_tensor, classic_group_entry_metadata_dict]
                        )
                else:
                    # L+G / L / G groups: use our self-contained per-stream code.
                    group_conditioning_entry = _encode_one_group_into_one_sdxl_conditioning_entry(
                        clip,
                        section_descriptors_in_this_group,
                        clip_pass_choice_for_this_group,
                        join_separator,
                        mask_token,
                        float(strict_mask),
                        float(start_from_masked),
                    )
                    group_conditioning_entry[1].update(sdxl_size_and_crop_metadata_fields)
                    output_conditioning_entries_combined_across_groups.append(group_conditioning_entry)
            except Exception as group_encoding_failure:
                logging.warning(
                    f"CLIPTextEncodeWithCutoffRegionSeparation: group '{clip_pass_choice_for_this_group}' "
                    f"encoding failed: {type(group_encoding_failure).__name__}: {group_encoding_failure}. "
                    f"Skipping this group."
                )
                continue

            full_prompt_text_for_this_group_display = _build_full_prompt_text_for_one_group_of_sections(
                section_descriptors_in_this_group, join_separator
            )
            per_group_full_prompt_text_for_reference_display.append(
                f"[{clip_pass_choice_for_this_group}] {full_prompt_text_for_this_group_display}"
            )

        reference_full_prompt_text_for_output = "\n".join(per_group_full_prompt_text_for_reference_display)
        return (output_conditioning_entries_combined_across_groups, reference_full_prompt_text_for_output)
