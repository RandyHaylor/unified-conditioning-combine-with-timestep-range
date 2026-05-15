"""
Conditioning Merge (with timestep ranges).

A single unified node that replaces the previous three (Concat / Combine /
Average). Features:

  - merge_mode dropdown: "concat" | "combine" | "average" | "average_normalized"
  - Dynamic 1..N CONDITIONING input slots (driven by web/conditioning_merge_with_timestep_ranges.js).
  - Each slot has three widgets: start, end, weight.
      - start/end intersect with that input's upstream start_percent/end_percent
        (widget=[0,1] is pass-through; widget can only narrow, never widen).
      - weight applies per-mode:
          concat              : torch.cat([weight_i * tokens_i for active], dim=1)
          combine             : each emitted entry gets metadata['strength'] = weight_i
                                (only written when weight != 1.0, to preserve stock semantics by default)
          average             : sum(weight_i * tokens_i for active)
          average_normalized  : sum(weight_i * tokens_i) / sum(weight_i)   over active subset

For concat / average / average_normalized, the [0,1] timestep span is segmented
at every active slot's effective-range endpoints; each non-empty sub-interval
emits one CONDITIONING entry. Stock samplers honor each entry's
start_percent/end_percent via comfy/samplers.py:calculate_start_end_timesteps.

Combine mode does not segment; it just emits each input's entries narrowed to
its effective range, with optional per-entry strength written.
"""

import logging
import re
from typing import Union

import torch

from ._timestep_range_helpers import (
    FLOAT_EPSILON_FOR_RANGE_BREAKPOINT_DEDUPE,
    compute_effective_range_from_widget_and_upstream,
    compute_sorted_unique_breakpoints,
    interval_is_inside_range,
)


CONDITIONING_SLOT_KEY_PATTERN = re.compile(r"^conditioning_(\d+)(_start|_end|_weight)?$")

MERGE_MODE_CONCAT = "concat"
MERGE_MODE_COMBINE = "combine"
MERGE_MODE_AVERAGE = "average"
MERGE_MODE_AVERAGE_NORMALIZED = "average_normalized"

MERGE_MODE_CHOICES_IN_DROPDOWN_ORDER = [
    MERGE_MODE_CONCAT,
    MERGE_MODE_COMBINE,
    MERGE_MODE_AVERAGE,
    MERGE_MODE_AVERAGE_NORMALIZED,
]


class FlexibleOptionalConditioningSlotInputType(dict):
    """
    Dict subclass that lets ComfyUI accept any extra `conditioning_<n>` /
    `_start` / `_end` / `_weight` keys at validation time. The frontend JS
    dynamically adds these inputs/widgets; the backend just has to declare it
    will accept them.

    Pattern reference (do NOT import from): rgthree-comfy's
    `FlexibleOptionalInputType` in custom_nodes/rgthree-comfy/py/utils.py.
    Reimplemented locally so we have zero third-party dependencies.
    """

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "merge_mode":
                return (MERGE_MODE_CHOICES_IN_DROPDOWN_ORDER, {"default": MERGE_MODE_CONCAT})
            match = CONDITIONING_SLOT_KEY_PATTERN.match(key)
            if match is not None:
                suffix = match.group(2)
                if suffix is None:
                    return ("CONDITIONING",)
                if suffix == "_start":
                    return ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001})
                if suffix == "_end":
                    return ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                if suffix == "_weight":
                    return ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})
        return ("*",)


def _zero_pad_or_truncate_tokens_to_match_reference_token_count(
    tokens_tensor_to_resize, reference_tokens_tensor
):
    reference_token_count = reference_tokens_tensor.shape[1]
    resized = tokens_tensor_to_resize[:, :reference_token_count]
    if resized.shape[1] < reference_token_count:
        resized = torch.cat(
            [resized]
            + [torch.zeros(
                (1, (reference_token_count - resized.shape[1]), reference_tokens_tensor.shape[2])
            )],
            dim=1,
        )
    return resized


def _parse_kwargs_into_per_slot_dict(kwargs):
    """Returns {slot_index: {'conditioning': list_or_None, 'start': float, 'end': float, 'weight': float}}."""
    per_slot_dict = {}
    for key, value in kwargs.items():
        match = CONDITIONING_SLOT_KEY_PATTERN.match(key)
        if match is None:
            continue
        slot_index = int(match.group(1))
        suffix = match.group(2)
        if slot_index not in per_slot_dict:
            per_slot_dict[slot_index] = {"conditioning": None, "start": 0.0, "end": 1.0, "weight": 1.0}
        if suffix is None:
            per_slot_dict[slot_index]["conditioning"] = value
        elif suffix == "_start":
            per_slot_dict[slot_index]["start"] = float(value)
        elif suffix == "_end":
            per_slot_dict[slot_index]["end"] = float(value)
        elif suffix == "_weight":
            per_slot_dict[slot_index]["weight"] = float(value)
    return per_slot_dict


def _flatten_connected_slots_into_per_entry_input_records(per_slot_dict):
    """
    For each connected slot, expand its CONDITIONING list of entries into individual
    input records keyed by (slot_index, entry_index_within_slot). Each record's effective
    range is the intersection of the slot's widget range with the entry's own upstream
    start_percent / end_percent (defaulting to [0,1] if absent).

    Returns list of dicts with keys:
      slot_index, entry_index, tokens_tensor, metadata_dict,
      effective_start, effective_end, weight
    """
    per_entry_input_records = []
    for slot_index in sorted(per_slot_dict.keys()):
        slot_data = per_slot_dict[slot_index]
        conditioning_list = slot_data["conditioning"]
        if conditioning_list is None:
            continue
        widget_start = slot_data["start"]
        widget_end = slot_data["end"]
        weight = slot_data["weight"]
        for entry_index_within_slot, conditioning_entry in enumerate(conditioning_list):
            entry_tokens_tensor = conditioning_entry[0]
            entry_metadata_dict = conditioning_entry[1]
            effective_start, effective_end = compute_effective_range_from_widget_and_upstream(
                widget_start, widget_end, entry_metadata_dict
            )
            if effective_end <= effective_start:
                continue
            per_entry_input_records.append({
                "slot_index": slot_index,
                "entry_index": entry_index_within_slot,
                "tokens_tensor": entry_tokens_tensor,
                "metadata_dict": entry_metadata_dict,
                "effective_start": effective_start,
                "effective_end": effective_end,
                "weight": weight,
            })
    return per_entry_input_records


def _merge_in_combine_mode(per_entry_input_records):
    output_conditioning_entries = []
    for record in per_entry_input_records:
        emitted_metadata_dict = record["metadata_dict"].copy()
        emitted_metadata_dict["start_percent"] = float(record["effective_start"])
        emitted_metadata_dict["end_percent"] = float(record["effective_end"])
        if record["weight"] != 1.0:
            emitted_metadata_dict["strength"] = float(record["weight"])
        output_conditioning_entries.append([record["tokens_tensor"], emitted_metadata_dict])
    return output_conditioning_entries


def _detect_overlapping_entries_within_any_single_slot(per_slot_dict):
    """
    Returns the first (slot_index, range_a, range_b) tuple where one slot has
    two entries whose effective timestep ranges overlap, or None if no slot
    has any overlapping entries.

    Overlapping entries on a single slot indicate upstream parallel-branch
    conditioning (typically the output of a 'combine' mode). Flat merge modes
    (concat / average / average_normalized) cannot represent parallel branches
    in their single-merged-segment output, so we refuse to run with such an
    input rather than silently flattening it.
    """
    for slot_index in sorted(per_slot_dict.keys()):
        slot_data = per_slot_dict[slot_index]
        conditioning_list = slot_data["conditioning"]
        if conditioning_list is None or len(conditioning_list) < 2:
            continue
        widget_start_for_slot = slot_data["start"]
        widget_end_for_slot = slot_data["end"]
        effective_ranges_for_each_entry_on_this_slot = []
        for conditioning_entry in conditioning_list:
            entry_metadata_dict = conditioning_entry[1]
            effective_start, effective_end = compute_effective_range_from_widget_and_upstream(
                widget_start_for_slot, widget_end_for_slot, entry_metadata_dict
            )
            if effective_end > effective_start:
                effective_ranges_for_each_entry_on_this_slot.append((effective_start, effective_end))
        for first_index_in_ranges_list in range(len(effective_ranges_for_each_entry_on_this_slot)):
            for second_index_in_ranges_list in range(first_index_in_ranges_list + 1, len(effective_ranges_for_each_entry_on_this_slot)):
                range_a_start, range_a_end = effective_ranges_for_each_entry_on_this_slot[first_index_in_ranges_list]
                range_b_start, range_b_end = effective_ranges_for_each_entry_on_this_slot[second_index_in_ranges_list]
                if max(range_a_start, range_b_start) < min(range_a_end, range_b_end):
                    return (
                        slot_index,
                        (range_a_start, range_a_end),
                        (range_b_start, range_b_end),
                    )
    return None


def _raise_value_error_if_flat_merge_mode_would_destroy_parallel_branches(per_slot_dict, merge_mode):
    overlap_detection_result = _detect_overlapping_entries_within_any_single_slot(per_slot_dict)
    if overlap_detection_result is None:
        return
    offending_slot_index, range_a, range_b = overlap_detection_result
    range_a_start, range_a_end = range_a
    range_b_start, range_b_end = range_b
    error_message_lines = [
        f"Unified Conditioning Merge (mode='{merge_mode}'): cannot accept this input.",
        "",
        f"Input on slot {offending_slot_index} is delivering MULTIPLE separate conditionings",
        f"that overlap in time:",
        f"  - one covers timesteps [{range_a_start:.4f} → {range_a_end:.4f}]",
        f"  - another covers timesteps [{range_b_start:.4f} → {range_b_end:.4f}]",
        "",
        "This shape of input is produced by 'combine' mode, which intentionally keeps the",
        "prompts as separate parallel branches so the sampler evaluates them independently",
        "and blends their predictions at each step.",
        "",
        f"The '{merge_mode}' mode you chose on THIS node would do the opposite: it would",
        "glue all the prompt tokens together into one single prompt per timestep segment.",
        "Doing that here would erase the upstream split entirely — your earlier 'combine'",
        "node would have had no effect on the image. We refuse rather than silently destroy",
        "that parallel structure.",
        "",
        "Three ways to fix this:",
        f"  1. Change THIS node's mode from '{merge_mode}' to 'combine'. The parallel",
        "     branches pass through unchanged and the sampler honors them.",
        "  2. Change the UPSTREAM node from 'combine' to 'concat' (or 'average' /",
        "     'average_normalized'). The earlier split is replaced with token-level",
        "     gluing, and this node then sees only one entry per slot.",
        f"  3. Restructure so no node downstream of a 'combine' uses '{merge_mode}'",
        "     mode — keep 'combine' all the way out to the sampler.",
    ]
    raise ValueError("\n".join(error_message_lines))


def _select_records_active_in_sub_interval(per_entry_input_records, sub_interval_start, sub_interval_end):
    return [
        record for record in per_entry_input_records
        if interval_is_inside_range(sub_interval_start, sub_interval_end, record["effective_start"], record["effective_end"])
    ]


def _concat_active_records_with_weights(active_records):
    weighted_token_tensors = [record["weight"] * record["tokens_tensor"] for record in active_records]
    return torch.cat(weighted_token_tensors, dim=1)


def _blend_active_records_with_weights(active_records, normalize_by_sum_of_weights):
    reference_tokens_tensor = active_records[0]["tokens_tensor"]
    sum_of_weights = sum(record["weight"] for record in active_records)
    if normalize_by_sum_of_weights:
        if sum_of_weights <= 0.0:
            normalization_divisor = 1.0
        else:
            normalization_divisor = sum_of_weights
    else:
        normalization_divisor = 1.0

    blended_tokens_tensor = torch.zeros_like(reference_tokens_tensor)
    for record in active_records:
        resized_tokens = _zero_pad_or_truncate_tokens_to_match_reference_token_count(
            record["tokens_tensor"], reference_tokens_tensor
        )
        blended_tokens_tensor = blended_tokens_tensor + (record["weight"] / normalization_divisor) * resized_tokens

    pooled_contributions = []
    for record in active_records:
        pooled = record["metadata_dict"].get("pooled_output", None)
        if pooled is not None:
            pooled_contributions.append((pooled, record["weight"]))
    blended_pooled_output = None
    if len(pooled_contributions) > 0:
        first_pooled, _ = pooled_contributions[0]
        blended_pooled_output = torch.zeros_like(first_pooled)
        for pooled, weight in pooled_contributions:
            blended_pooled_output = blended_pooled_output + (weight / normalization_divisor) * pooled

    return blended_tokens_tensor, blended_pooled_output


def _merge_in_segmented_mode(per_entry_input_records, merge_mode):
    output_conditioning_entries = []

    all_range_endpoints = []
    for record in per_entry_input_records:
        all_range_endpoints.append(record["effective_start"])
        all_range_endpoints.append(record["effective_end"])
    breakpoints = compute_sorted_unique_breakpoints(*all_range_endpoints)

    for breakpoint_index in range(len(breakpoints) - 1):
        sub_interval_start = breakpoints[breakpoint_index]
        sub_interval_end = breakpoints[breakpoint_index + 1]
        if (sub_interval_end - sub_interval_start) <= FLOAT_EPSILON_FOR_RANGE_BREAKPOINT_DEDUPE:
            continue

        active_records = _select_records_active_in_sub_interval(
            per_entry_input_records, sub_interval_start, sub_interval_end
        )
        if len(active_records) == 0:
            continue

        if merge_mode == MERGE_MODE_CONCAT:
            emitted_tokens_tensor = _concat_active_records_with_weights(active_records)
            emitted_metadata_dict = active_records[0]["metadata_dict"].copy()
        elif merge_mode == MERGE_MODE_AVERAGE:
            emitted_tokens_tensor, blended_pooled_output = _blend_active_records_with_weights(
                active_records, normalize_by_sum_of_weights=False
            )
            emitted_metadata_dict = active_records[0]["metadata_dict"].copy()
            if blended_pooled_output is not None:
                emitted_metadata_dict["pooled_output"] = blended_pooled_output
        elif merge_mode == MERGE_MODE_AVERAGE_NORMALIZED:
            emitted_tokens_tensor, blended_pooled_output = _blend_active_records_with_weights(
                active_records, normalize_by_sum_of_weights=True
            )
            emitted_metadata_dict = active_records[0]["metadata_dict"].copy()
            if blended_pooled_output is not None:
                emitted_metadata_dict["pooled_output"] = blended_pooled_output
        else:
            raise ValueError(f"Unknown merge_mode '{merge_mode}'")

        emitted_metadata_dict["start_percent"] = float(sub_interval_start)
        emitted_metadata_dict["end_percent"] = float(sub_interval_end)
        output_conditioning_entries.append([emitted_tokens_tensor, emitted_metadata_dict])

    return output_conditioning_entries


class ConditioningMergeWithTimestepRanges:
    @classmethod
    def INPUT_TYPES(cls):
        # Slot 1's start/end/weight widgets are NOT in `required` — the JS
        # extension creates them dynamically only when slot 1 is connected,
        # so that an empty slot 1 shows no widgets (consistent with how empty
        # slots 2+ behave).
        return {
            "required": {
                "merge_mode": (MERGE_MODE_CHOICES_IN_DROPDOWN_ORDER, {"default": MERGE_MODE_CONCAT}),
                "conditioning_1": ("CONDITIONING",),
            },
            "optional": FlexibleOptionalConditioningSlotInputType(),
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "merge_with_timestep_ranges"
    CATEGORY = "unified-conditioning-merge"

    def merge_with_timestep_ranges(self, **kwargs):
        merge_mode = kwargs.get("merge_mode", MERGE_MODE_CONCAT)

        per_slot_dict = _parse_kwargs_into_per_slot_dict(kwargs)
        per_entry_input_records = _flatten_connected_slots_into_per_entry_input_records(per_slot_dict)

        if len(per_entry_input_records) == 0:
            logging.warning("ConditioningMergeWithTimestepRanges: no connected inputs with non-empty effective ranges; returning empty list.")
            return ([],)

        if merge_mode == MERGE_MODE_COMBINE:
            output_conditioning_entries = _merge_in_combine_mode(per_entry_input_records)
        else:
            _raise_value_error_if_flat_merge_mode_would_destroy_parallel_branches(per_slot_dict, merge_mode)
            output_conditioning_entries = _merge_in_segmented_mode(per_entry_input_records, merge_mode)

        return (output_conditioning_entries,)
