from .conditioning_merge_with_timestep_ranges import ConditioningMergeWithTimestepRanges
from .debug_conditioning import DebugConditioning

NODE_CLASS_MAPPINGS = {
    "ConditioningMergeWithTimestepRanges": ConditioningMergeWithTimestepRanges,
    "DebugConditioning": DebugConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningMergeWithTimestepRanges": "Unified Conditioning Merge (with timestep ranges)",
    "DebugConditioning": "Debug Conditioning",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
