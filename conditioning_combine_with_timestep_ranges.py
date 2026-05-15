"""
Conditioning Combine (with timestep ranges).

Stock ConditioningCombine is already lossless (it list-concatenates entries, so
each side's per-entry start_percent/end_percent survives). This variant adds
widget-controlled range narrowing on each side:

  - Widget range INTERSECTS with each entry's upstream start_percent/end_percent.
  - Widget [0,1] is a pass-through (identical to stock Combine).
  - Both inputs may be multi-entry; every entry on both sides is emitted with
    its effective (intersected) range. Entries whose effective range is empty
    are dropped.
"""

from ._timestep_range_helpers import compute_effective_range_from_widget_and_upstream


class ConditioningCombineWithTimestepRanges:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning_1": ("CONDITIONING",),
                "conditioning_2": ("CONDITIONING",),
                "conditioning_1_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "conditioning_1_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "conditioning_2_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "conditioning_2_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine_with_timestep_ranges"
    CATEGORY = "advanced/conditioning"

    def combine_with_timestep_ranges(
        self,
        conditioning_1,
        conditioning_2,
        conditioning_1_start,
        conditioning_1_end,
        conditioning_2_start,
        conditioning_2_end,
    ):
        output_conditioning_entries = []

        for source_conditioning_list, widget_start_value, widget_end_value in (
            (conditioning_1, conditioning_1_start, conditioning_1_end),
            (conditioning_2, conditioning_2_start, conditioning_2_end),
        ):
            for source_entry in source_conditioning_list:
                source_tokens_tensor = source_entry[0]
                source_metadata_dict = source_entry[1]

                effective_start, effective_end = compute_effective_range_from_widget_and_upstream(
                    widget_start_value, widget_end_value, source_metadata_dict
                )

                if effective_end <= effective_start:
                    continue

                emitted_metadata_dict = source_metadata_dict.copy()
                emitted_metadata_dict["start_percent"] = float(effective_start)
                emitted_metadata_dict["end_percent"] = float(effective_end)

                output_conditioning_entries.append([source_tokens_tensor, emitted_metadata_dict])

        return (output_conditioning_entries,)
