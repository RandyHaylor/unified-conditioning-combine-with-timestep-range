from .conditioning_concat_with_timestep_ranges import ConditioningConcatWithTimestepRanges
from .conditioning_combine_with_timestep_ranges import ConditioningCombineWithTimestepRanges
from .conditioning_average_with_timestep_ranges import ConditioningAverageWithTimestepRanges

NODE_CLASS_MAPPINGS = {
    "ConditioningConcatWithTimestepRanges": ConditioningConcatWithTimestepRanges,
    "ConditioningCombineWithTimestepRanges": ConditioningCombineWithTimestepRanges,
    "ConditioningAverageWithTimestepRanges": ConditioningAverageWithTimestepRanges,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningConcatWithTimestepRanges": "Conditioning Concat (with timestep ranges)",
    "ConditioningCombineWithTimestepRanges": "Conditioning Combine (with timestep ranges)",
    "ConditioningAverageWithTimestepRanges": "Conditioning Average (with timestep ranges)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
