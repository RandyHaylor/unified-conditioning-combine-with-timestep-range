"""
CLIP Text Encode (Cutoff Region Separation)

Single-node prompt builder that combines:
  - Phrase-level decontamination via BlenderNeko's ComfyUI_Cutoff
  - Per-section CLIP-stream routing (Pass L+G / Pass L / Pass G) for SDXL
  - SDXL size/crop metadata (zoom + x/y offsets) applied to every output
    entry, with target W/H auto-derived from an optional LATENT input
    (defaults to 1024x1024 when LATENT is not connected)

Algorithm (Option A: group-by-stream cutoff):
  1. Collect active sections (non-empty text), each with isolate flag,
     weight, and CLIP-stream-pass dropdown choice.
  2. Group sections by their clip pass choice into:
        L+G group, L-only group, G-only group.
  3. For each non-empty group:
       a. Build the group's full prompt = join_separator.join(group texts).
          (Per-section weight on non-isolate sections is applied via
          CLIP `(text:weight)` attention syntax in the joined text.)
       b. Run ComfyUI_Cutoff on that group:
          init_prompt → add_clip_region for each isolate=True section →
          finalize_clip_regions.
       c. Mask the resulting tokens tensor according to the group's
          stream choice:
            - L+G: no mask (full SDXL tensor)
            - L only: zero [:, :, 768:]  (CLIP-G portion) + zero pooled
            - G only: zero [:, :, :768]  (CLIP-L portion)
          (Mask only applied when last embedding dim == 2048, the SDXL
          shape; other shapes pass through unchanged.)
       d. Stamp SDXL size/crop metadata onto every entry of this group's
          CONDITIONING.
  4. Combine the per-group CONDITIONINGs into one multi-entry CONDITIONING
     output (combine-style — sampler treats each group as a parallel
     branch). If only one group is populated, output is single-entry.

Requires ComfyUI_Cutoff installed
(https://github.com/BlenderNeko/ComfyUI_Cutoff).
"""

import inspect
import logging
import re
import sys

import torch


MAX_SECTION_COUNT_SUPPORTED = 16
DEFAULT_SECTION_COUNT_VALUE = 3

CLIP_STREAM_PASS_BOTH_L_AND_G = "Pass L+G"
CLIP_STREAM_PASS_L_ONLY = "Pass L"
CLIP_STREAM_PASS_G_ONLY = "Pass G"
CLIP_STREAM_PASS_CHOICES_IN_DROPDOWN_ORDER = [
    CLIP_STREAM_PASS_BOTH_L_AND_G,
    CLIP_STREAM_PASS_L_ONLY,
    CLIP_STREAM_PASS_G_ONLY,
]

# SDXL token-embedding layout in the CONDITIONING tensor's last dim.
SDXL_EMBEDDING_TOTAL_DIM = 2048
SDXL_CLIP_L_PORTION_DIM = 768

# Latent → image-space spatial scale factor (standard VAE downscale of 8).
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


# -------------- cutoff discovery --------------

def _find_loaded_cutoff_module_in_sys_modules_or_none():
    """
    Returns the ComfyUI_Cutoff plugin's `cutoff` module by scanning sys.modules
    for one that exposes a callable `finalize_clip_regions` and a real Python
    class `CLIPRegionsBasePrompt`. Prefers modules whose name contains
    "cutoff" (case-insensitive).
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
        if not callable(finalize_attribute_or_none):
            continue
        if not inspect.isclass(base_prompt_attribute_or_none):
            continue
        matching_candidate_modules_keyed_by_name[module_name_in_sys_modules] = loaded_module_object

    if not matching_candidate_modules_keyed_by_name:
        return None
    for candidate_module_name, candidate_module in matching_candidate_modules_keyed_by_name.items():
        if "cutoff" in candidate_module_name.lower():
            return candidate_module
    return next(iter(matching_candidate_modules_keyed_by_name.values()))


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
    """
    Returns {clip_pass_choice: [section_descriptor, ...]} preserving original
    section order within each group.
    """
    grouped_sections_by_clip_pass_choice = {}
    for section_descriptor in active_section_descriptors_list:
        clip_pass_choice_for_this_section = section_descriptor["clip_pass_choice"]
        if clip_pass_choice_for_this_section not in grouped_sections_by_clip_pass_choice:
            grouped_sections_by_clip_pass_choice[clip_pass_choice_for_this_section] = []
        grouped_sections_by_clip_pass_choice[clip_pass_choice_for_this_section].append(section_descriptor)
    return grouped_sections_by_clip_pass_choice


# -------------- cutoff per group --------------

def _build_populated_cutoff_clip_regions_state_for_one_group(
    clip_object, full_prompt_text_for_group, section_descriptors_in_group_list, cutoff_module
):
    base_prompt_node_instance = cutoff_module.CLIPRegionsBasePrompt()
    base_state_tuple = base_prompt_node_instance.init_prompt(clip_object, full_prompt_text_for_group)
    current_clip_regions_state = base_state_tuple[0]

    add_region_node_instance = cutoff_module.CLIPSetRegion()
    for section_descriptor in section_descriptors_in_group_list:
        if not section_descriptor["isolate"]:
            continue
        try:
            next_state_tuple = add_region_node_instance.add_clip_region(
                current_clip_regions_state,
                section_descriptor["text"],
                section_descriptor["text"],
                section_descriptor["weight"],
            )
            current_clip_regions_state = next_state_tuple[0]
        except Exception as cutoff_register_region_exception:
            logging.warning(
                "CLIPTextEncodeWithCutoffRegionSeparation: skipped isolate "
                f"registration for section text {section_descriptor['text']!r} "
                f"(Cutoff raised {type(cutoff_register_region_exception).__name__}: "
                f"{cutoff_register_region_exception})."
            )
    return current_clip_regions_state


# -------------- clip stream masking --------------

def _apply_clip_pass_choice_mask_to_tokens_and_pooled_in_one_conditioning_entry(
    conditioning_entry, clip_pass_choice
):
    """
    Returns a new [tokens_tensor, metadata_dict] with the L or G portion of
    the tokens tensor zeroed per clip_pass_choice. Pass L+G returns the
    entry unchanged. Only acts when last embedding dim == 2048 (SDXL).
    """
    original_tokens_tensor = conditioning_entry[0]
    original_metadata_dict = conditioning_entry[1]

    if clip_pass_choice == CLIP_STREAM_PASS_BOTH_L_AND_G:
        return [original_tokens_tensor, dict(original_metadata_dict)]

    last_axis_size = original_tokens_tensor.shape[-1] if original_tokens_tensor is not None else 0
    if last_axis_size != SDXL_EMBEDDING_TOTAL_DIM:
        return [original_tokens_tensor, dict(original_metadata_dict)]

    masked_tokens_tensor = original_tokens_tensor.clone()
    masked_metadata_dict = dict(original_metadata_dict)
    if clip_pass_choice == CLIP_STREAM_PASS_L_ONLY:
        masked_tokens_tensor[:, :, SDXL_CLIP_L_PORTION_DIM:] = 0.0
        existing_pooled_output = masked_metadata_dict.get("pooled_output", None)
        if existing_pooled_output is not None:
            masked_metadata_dict["pooled_output"] = torch.zeros_like(existing_pooled_output)
    elif clip_pass_choice == CLIP_STREAM_PASS_G_ONLY:
        masked_tokens_tensor[:, :, :SDXL_CLIP_L_PORTION_DIM] = 0.0

    return [masked_tokens_tensor, masked_metadata_dict]


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


def _apply_metadata_fields_to_each_entry_of_one_conditioning(input_conditioning_list, metadata_fields_to_merge):
    output_conditioning_entries = []
    for conditioning_entry in input_conditioning_list:
        entry_tokens_tensor = conditioning_entry[0]
        entry_metadata_dict = conditioning_entry[1]
        updated_metadata_dict = dict(entry_metadata_dict)
        updated_metadata_dict.update(metadata_fields_to_merge)
        output_conditioning_entries.append([entry_tokens_tensor, updated_metadata_dict])
    return output_conditioning_entries


# -------------- flexible-input dict for the per-section widgets --------------

CONDITIONING_SLOT_KEY_PATTERN = re.compile(r"^section_(\d+)_(text|isolate|weight|clip)$")


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
    FUNCTION = "encode_with_grouped_cutoff_and_stream_masking_and_sdxl_zoom"
    CATEGORY = "unified-conditioning-merge"

    def encode_with_grouped_cutoff_and_stream_masking_and_sdxl_zoom(
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
        cutoff_module = _find_loaded_cutoff_module_in_sys_modules_or_none()
        if cutoff_module is None:
            raise RuntimeError(
                "CLIPTextEncodeWithCutoffRegionSeparation: the ComfyUI_Cutoff plugin is "
                "required but was not found in loaded modules. Install it from "
                "https://github.com/BlenderNeko/ComfyUI_Cutoff into ComfyUI/custom_nodes/ "
                "and restart ComfyUI."
            )

        active_section_descriptors_list = _collect_active_non_empty_sections_from_kwargs(
            kwargs_for_individual_section_widget_values, section_count
        )

        # Resolve target image W/H from optional LATENT, default 1024x1024.
        target_image_width, target_image_height = _resolve_target_image_width_and_height_from_optional_latent_or_defaults(latent)
        zoom_factor_clamped = max(ZOOM_MINIMUM_VALUE, float(zoom))
        offset_x_clamped = _clamp_numeric_value_inclusive(float(offset_x), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE)
        offset_y_clamped = _clamp_numeric_value_inclusive(float(offset_y), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE)
        sdxl_size_and_crop_metadata_fields = _compute_sdxl_size_and_crop_metadata_fields(
            target_image_width, target_image_height, zoom_factor_clamped, offset_x_clamped, offset_y_clamped
        )

        if not active_section_descriptors_list:
            base_prompt_node_instance = cutoff_module.CLIPRegionsBasePrompt()
            empty_state_tuple = base_prompt_node_instance.init_prompt(clip, "")
            empty_finalize_tuple = cutoff_module.finalize_clip_regions(
                empty_state_tuple[0], mask_token, float(strict_mask), float(start_from_masked)
            )
            empty_conditioning_with_sdxl_metadata = _apply_metadata_fields_to_each_entry_of_one_conditioning(
                empty_finalize_tuple[0], sdxl_size_and_crop_metadata_fields
            )
            return (empty_conditioning_with_sdxl_metadata, "")

        # Group sections by their CLIP stream pass choice.
        grouped_sections_by_clip_pass_choice = _group_active_sections_by_their_clip_pass_choice(
            active_section_descriptors_list
        )

        # Process each non-empty group through cutoff, mask result per stream
        # choice, stamp SDXL metadata. Concatenate all output entries into one
        # multi-entry CONDITIONING (combine-style).
        output_conditioning_entries_combined_across_groups = []
        per_group_full_prompt_text_for_reference_display = []
        for clip_pass_choice_for_this_group, section_descriptors_in_this_group in grouped_sections_by_clip_pass_choice.items():
            full_prompt_text_for_this_group = _build_full_prompt_text_for_one_group_of_sections(
                section_descriptors_in_this_group, join_separator
            )
            if not full_prompt_text_for_this_group:
                continue
            populated_clip_regions_state_for_this_group = _build_populated_cutoff_clip_regions_state_for_one_group(
                clip, full_prompt_text_for_this_group, section_descriptors_in_this_group, cutoff_module
            )
            finalize_return_tuple_for_this_group = cutoff_module.finalize_clip_regions(
                populated_clip_regions_state_for_this_group,
                mask_token,
                float(strict_mask),
                float(start_from_masked),
            )
            unmasked_conditioning_for_this_group = finalize_return_tuple_for_this_group[0]
            for conditioning_entry_for_this_group in unmasked_conditioning_for_this_group:
                stream_masked_conditioning_entry = _apply_clip_pass_choice_mask_to_tokens_and_pooled_in_one_conditioning_entry(
                    conditioning_entry_for_this_group, clip_pass_choice_for_this_group
                )
                # Stamp SDXL metadata onto the entry.
                stream_masked_entry_tokens_tensor = stream_masked_conditioning_entry[0]
                stream_masked_entry_metadata_dict = dict(stream_masked_conditioning_entry[1])
                stream_masked_entry_metadata_dict.update(sdxl_size_and_crop_metadata_fields)
                output_conditioning_entries_combined_across_groups.append(
                    [stream_masked_entry_tokens_tensor, stream_masked_entry_metadata_dict]
                )
            per_group_full_prompt_text_for_reference_display.append(
                f"[{clip_pass_choice_for_this_group}] {full_prompt_text_for_this_group}"
            )

        reference_full_prompt_text_for_output = "\n".join(per_group_full_prompt_text_for_reference_display)
        return (output_conditioning_entries_combined_across_groups, reference_full_prompt_text_for_output)
