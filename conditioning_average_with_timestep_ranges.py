"""
Conditioning Average (with timestep ranges).

Stock ConditioningAverage preserves only the `to` side's metadata dict, so a
SetTimestepRange placed on the `from` side is silently discarded.

This node exposes start/end timestep ranges for BOTH inputs. Widget ranges
INTERSECT with any upstream start_percent/end_percent already on each entry's
metadata dict, so widget=[0,1] is a pass-through and the node can only narrow
ranges, never widen them.

For each (conditioning_to[i], conditioning_from[0]) pair, the effective ranges
are segmented at all breakpoints and emitted as:
  - Sub-intervals where only `to` is active emit `to` tokens unchanged.
  - Sub-intervals where only `from` is active emit `from` tokens unchanged.
  - Sub-intervals where both are active emit the weighted blend
    (same tensor math and pooled-output blend as stock ConditioningAverage,
    including the zero-pad / truncate of `from` to match `to`'s token count).
  - Sub-intervals where neither is active emit nothing.
"""

import torch

from ._timestep_range_helpers import (
    FLOAT_EPSILON_FOR_RANGE_BREAKPOINT_DEDUPE,
    compute_effective_range_from_widget_and_upstream,
    compute_sorted_unique_breakpoints,
    interval_is_inside_range,
)


def _compute_weighted_blend_of_tokens_and_pooled_matching_stock_average(
    to_tokens_tensor,
    to_metadata_dict,
    from_tokens_tensor,
    from_metadata_dict,
    conditioning_to_strength,
):
    pooled_output_from = from_metadata_dict.get("pooled_output", None)
    pooled_output_to = to_metadata_dict.get("pooled_output", pooled_output_from)

    from_tokens_truncated_or_padded = from_tokens_tensor[:, :to_tokens_tensor.shape[1]]
    if from_tokens_truncated_or_padded.shape[1] < to_tokens_tensor.shape[1]:
        from_tokens_truncated_or_padded = torch.cat(
            [from_tokens_truncated_or_padded]
            + [torch.zeros(
                (1, (to_tokens_tensor.shape[1] - from_tokens_truncated_or_padded.shape[1]), to_tokens_tensor.shape[2])
            )],
            dim=1,
        )

    blended_tokens_tensor = (
        torch.mul(to_tokens_tensor, conditioning_to_strength)
        + torch.mul(from_tokens_truncated_or_padded, (1.0 - conditioning_to_strength))
    )

    emitted_metadata_dict = to_metadata_dict.copy()
    if pooled_output_from is not None and pooled_output_to is not None:
        emitted_metadata_dict["pooled_output"] = (
            torch.mul(pooled_output_to, conditioning_to_strength)
            + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
        )
    elif pooled_output_from is not None:
        emitted_metadata_dict["pooled_output"] = pooled_output_from

    return blended_tokens_tensor, emitted_metadata_dict


class ConditioningAverageWithTimestepRanges:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_to": ("CONDITIONING",),
                "conditioning_from": ("CONDITIONING",),
                "conditioning_to_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "to_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "to_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "from_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "from_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "average_with_timestep_ranges"
    CATEGORY = "advanced/conditioning"

    def average_with_timestep_ranges(
        self,
        conditioning_to,
        conditioning_from,
        conditioning_to_strength,
        to_start,
        to_end,
        from_start,
        from_end,
    ):
        if len(conditioning_from) > 1:
            import logging
            logging.warning(
                "ConditioningAverageWithTimestepRanges: conditioning_from contains more than 1 entry; only the first will be used (matches stock ConditioningAverage behavior)."
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
                    emitted_tokens_tensor, emitted_metadata_dict = _compute_weighted_blend_of_tokens_and_pooled_matching_stock_average(
                        to_tokens_tensor,
                        to_metadata_dict,
                        from_tokens_tensor,
                        from_metadata_dict,
                        conditioning_to_strength,
                    )
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
