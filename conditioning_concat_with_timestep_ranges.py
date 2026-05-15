"""
Conditioning Concat (with timestep ranges).

Stock ComfyUI ConditioningConcat preserves only the `to` side's metadata dict,
so a SetTimestepRange placed on the `from` side is silently discarded.

This node exposes start/end timestep ranges for BOTH inputs. Widget ranges
INTERSECT with any upstream start_percent/end_percent already on each entry's
metadata dict, so widget=[0,1] is a pass-through and the node can only narrow
ranges, never widen them.

For each (conditioning_to[i], conditioning_from[0]) pair, the effective ranges
are segmented at all breakpoints and emitted as:
  - Sub-intervals where only `to` is active emit `to` tokens.
  - Sub-intervals where only `from` is active emit `from` tokens.
  - Sub-intervals where both are active emit token-concat(to, from)
    (same tensor-concat as stock ConditioningConcat).
  - Sub-intervals where neither is active emit nothing.
"""

import torch

from ._timestep_range_helpers import (
    FLOAT_EPSILON_FOR_RANGE_BREAKPOINT_DEDUPE,
    compute_effective_range_from_widget_and_upstream,
    compute_sorted_unique_breakpoints,
    interval_is_inside_range,
)


class ConditioningConcatWithTimestepRanges:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_to": ("CONDITIONING",),
                "conditioning_from": ("CONDITIONING",),
                "to_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "to_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "from_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "from_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "concat_with_timestep_ranges"
    CATEGORY = "advanced/conditioning"

    def concat_with_timestep_ranges(
        self,
        conditioning_to,
        conditioning_from,
        to_start,
        to_end,
        from_start,
        from_end,
    ):
        if len(conditioning_from) > 1:
            import logging
            logging.warning(
                "ConditioningConcatWithTimestepRanges: conditioning_from contains more than 1 entry; only the first will be used (matches stock ConditioningConcat behavior)."
            )

        from_tokens_tensor = conditioning_from[0][0]
        from_metadata_dict = conditioning_from[0][1]

        effective_from_start, effective_from_end = compute_effective_range_from_widget_and_upstream(
            from_start, from_end, from_metadata_dict
        )

        output_conditioning_entries = []

        for entry_index_in_to in range(len(conditioning_to)):
            to_tokens_tensor = conditioning_to[entry_index_in_to][0]
            to_metadata_dict = conditioning_to[entry_index_in_to][1]

            effective_to_start, effective_to_end = compute_effective_range_from_widget_and_upstream(
                to_start, to_end, to_metadata_dict
            )

            breakpoints = compute_sorted_unique_breakpoints(
                effective_to_start, effective_to_end, effective_from_start, effective_from_end
            )

            for breakpoint_index in range(len(breakpoints) - 1):
                sub_interval_start = breakpoints[breakpoint_index]
                sub_interval_end = breakpoints[breakpoint_index + 1]

                if (sub_interval_end - sub_interval_start) <= FLOAT_EPSILON_FOR_RANGE_BREAKPOINT_DEDUPE:
                    continue

                to_is_active_in_sub_interval = interval_is_inside_range(sub_interval_start, sub_interval_end, effective_to_start, effective_to_end)
                from_is_active_in_sub_interval = interval_is_inside_range(sub_interval_start, sub_interval_end, effective_from_start, effective_from_end)

                if not to_is_active_in_sub_interval and not from_is_active_in_sub_interval:
                    continue

                if to_is_active_in_sub_interval and from_is_active_in_sub_interval:
                    emitted_tokens_tensor = torch.cat((to_tokens_tensor, from_tokens_tensor), dim=1)
                    emitted_metadata_dict = to_metadata_dict.copy()
                elif to_is_active_in_sub_interval:
                    emitted_tokens_tensor = to_tokens_tensor
                    emitted_metadata_dict = to_metadata_dict.copy()
                else:
                    emitted_tokens_tensor = from_tokens_tensor
                    emitted_metadata_dict = from_metadata_dict.copy()

                emitted_metadata_dict["start_percent"] = float(sub_interval_start)
                emitted_metadata_dict["end_percent"] = float(sub_interval_end)

                output_conditioning_entries.append([emitted_tokens_tensor, emitted_metadata_dict])

        return (output_conditioning_entries,)
