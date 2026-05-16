"""
Conditioning Cutoff Sections Prompt

Single-node prompt builder that uses the Cutoff algorithm (from BlenderNeko's
ComfyUI_Cutoff) to confine each section's contextual influence to its own
tokens — preventing prompt context bleeding between sections.

UI shape:
  - section_count INT (1..16): how many section pairs are active.
  - global settings (join_separator, mask_token, strict_mask, start_from_masked).
  - Up to MAX_SECTION_COUNT_SUPPORTED pairs of (section_N_text,
    section_N_isolate) widgets. Pairs with index > section_count are
    hidden by the frontend JS extension; their values still serialize so
    you can re-expand without losing typed text.

Execution:
  - Build full_prompt = join_separator.join(text of active non-empty sections).
  - For each active section with isolate=True, register region_text =
    target_text = section text (weight=1.0) — phrase-level decontamination.
  - Finalize through ComfyUI_Cutoff's finalize_clip_regions and return
    the resulting CONDITIONING plus the reference full prompt text.

Requires ComfyUI_Cutoff installed
(https://github.com/BlenderNeko/ComfyUI_Cutoff). If missing, raises a
clear runtime error pointing at the install URL.
"""

import inspect
import logging
import re
import sys


WHITESPACE_RUN_REGEX_FOR_NORMALIZING_SECTION_TEXT = re.compile(r"\s+")


MAX_SECTION_COUNT_SUPPORTED = 16
DEFAULT_SECTION_COUNT_VALUE = 3


def _find_loaded_cutoff_module_in_sys_modules_or_none():
    """
    Returns the ComfyUI_Cutoff plugin's `cutoff` module by scanning sys.modules
    for one that exposes BOTH:
      - a callable `finalize_clip_regions`
      - a real Python class `CLIPRegionsBasePrompt` (verified via inspect.isclass)

    The class check is critical — without it, modules like `torch._OpNamespace`
    whose `__getattr__` returns op handles for any requested attribute name
    will pass a naive hasattr check, leading to TypeErrors when we try to
    instantiate the returned non-class.

    Prefers modules whose name contains "cutoff" (case-insensitive) when
    multiple candidates match.
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

    # Prefer modules whose name contains "cutoff" (case-insensitive).
    for candidate_module_name, candidate_module in matching_candidate_modules_keyed_by_name.items():
        if "cutoff" in candidate_module_name.lower():
            return candidate_module

    # Fallback: first match in iteration order.
    return next(iter(matching_candidate_modules_keyed_by_name.values()))


def _collect_active_non_empty_sections_from_kwargs(kwargs_dict, active_section_count):
    """
    Walks the section_N_text / section_N_isolate keys for N in 1..active_section_count
    and returns a list of {"text": stripped_text, "isolate": bool} dicts,
    skipping sections whose stripped text is empty.
    """
    active_section_descriptors_list = []
    for section_index in range(1, int(active_section_count) + 1):
        section_text_raw = kwargs_dict.get(f"section_{section_index}_text", "")
        section_isolate_raw = kwargs_dict.get(f"section_{section_index}_isolate", True)
        section_weight_raw = kwargs_dict.get(f"section_{section_index}_weight", 1.0)
        # Collapse any whitespace run (multiple spaces, tabs, newlines) into
        # a single space, then strip. This avoids Cutoff's
        # target_text.split(" ") producing empty strings (e.g. "warm  light"
        # split on space gives ["warm", "", "light"]), which would tokenize
        # to an empty list and trip an IndexError inside Cutoff's
        # get_sublists at sub_list[0].
        section_text_normalized_whitespace = WHITESPACE_RUN_REGEX_FOR_NORMALIZING_SECTION_TEXT.sub(
            " ", (section_text_raw or "")
        ).strip()
        if not section_text_normalized_whitespace:
            continue
        active_section_descriptors_list.append({
            "text": section_text_normalized_whitespace,
            "isolate": bool(section_isolate_raw),
            "weight": float(section_weight_raw),
        })
    return active_section_descriptors_list


def _build_full_prompt_text_from_section_descriptors_using_separator(
    section_descriptors_list, join_separator_string
):
    return join_separator_string.join(descriptor["text"] for descriptor in section_descriptors_list)


def _build_populated_cutoff_clip_regions_state_with_isolated_sections_registered(
    clip_object, full_prompt_text, section_descriptors_list, cutoff_module
):
    base_prompt_node_instance = cutoff_module.CLIPRegionsBasePrompt()
    base_state_tuple = base_prompt_node_instance.init_prompt(clip_object, full_prompt_text)
    current_clip_regions_state = base_state_tuple[0]

    add_region_node_instance = cutoff_module.CLIPSetRegion()
    for section_descriptor in section_descriptors_list:
        if not section_descriptor["isolate"]:
            continue
        # region_text == target_text == section text gives "phrase-level
        # decontamination" — confines that whole phrase's influence to its
        # own token region.
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
                "ConditioningCutoffSectionsPrompt: skipped isolate registration "
                f"for section text {section_descriptor['text']!r} "
                f"(Cutoff raised {type(cutoff_register_region_exception).__name__}: "
                f"{cutoff_register_region_exception}). Section is still in the "
                "full prompt but not isolated."
            )
    return current_clip_regions_state


class ConditioningCutoffSectionsPrompt:
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
        return {"required": required_inputs_dict}

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "reference_full_prompt")
    FUNCTION = "build_cutoff_conditioning_from_active_sections"
    CATEGORY = "unified-conditioning-merge"

    def build_cutoff_conditioning_from_active_sections(
        self,
        clip,
        section_count,
        join_separator,
        mask_token,
        strict_mask,
        start_from_masked,
        **kwargs_for_individual_section_widget_values,
    ):
        cutoff_module = _find_loaded_cutoff_module_in_sys_modules_or_none()
        if cutoff_module is None:
            raise RuntimeError(
                "ConditioningCutoffSectionsPrompt: the ComfyUI_Cutoff plugin is required "
                "but was not found in loaded modules. Install it from "
                "https://github.com/BlenderNeko/ComfyUI_Cutoff into ComfyUI/custom_nodes/ "
                "and restart ComfyUI."
            )

        active_section_descriptors_list = _collect_active_non_empty_sections_from_kwargs(
            kwargs_for_individual_section_widget_values, section_count
        )

        full_prompt_text_for_clip_to_see = _build_full_prompt_text_from_section_descriptors_using_separator(
            active_section_descriptors_list, join_separator
        )

        if not active_section_descriptors_list:
            # No content — encode empty prompt directly so we still return
            # a valid CONDITIONING entry.
            base_prompt_node_instance = cutoff_module.CLIPRegionsBasePrompt()
            empty_state_tuple = base_prompt_node_instance.init_prompt(clip, "")
            empty_finalize_tuple = cutoff_module.finalize_clip_regions(
                empty_state_tuple[0], mask_token, float(strict_mask), float(start_from_masked)
            )
            return (empty_finalize_tuple[0], "")

        populated_clip_regions_state = _build_populated_cutoff_clip_regions_state_with_isolated_sections_registered(
            clip, full_prompt_text_for_clip_to_see, active_section_descriptors_list, cutoff_module
        )

        finalize_return_tuple = cutoff_module.finalize_clip_regions(
            populated_clip_regions_state,
            mask_token,
            float(strict_mask),
            float(start_from_masked),
        )

        return (finalize_return_tuple[0], full_prompt_text_for_clip_to_see)
