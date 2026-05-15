from .conditioning_merge_with_timestep_ranges import ConditioningMergeWithTimestepRanges

NODE_CLASS_MAPPINGS = {
    "ConditioningMergeWithTimestepRanges": ConditioningMergeWithTimestepRanges,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningMergeWithTimestepRanges": "Conditioning Merge (with timestep ranges)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
