from .conditioning_merge_with_timestep_ranges import ConditioningMergeWithTimestepRanges
from .debug_conditioning import DebugConditioning
from .clip_text_encode_sdxl_auto_split_and_merge import CLIPTextEncodeSDXLAutoSplitAndMerge
from .conditioning_crop_zoom_sdxl import ConditioningCropZoomSDXL
from .conditioning_cutoff_sections_prompt import ConditioningCutoffSectionsPrompt

NODE_CLASS_MAPPINGS = {
    "ConditioningMergeWithTimestepRanges": ConditioningMergeWithTimestepRanges,
    "DebugConditioning": DebugConditioning,
    "CLIPTextEncodeSDXLAutoSplitAndMerge": CLIPTextEncodeSDXLAutoSplitAndMerge,
    "ConditioningCropZoomSDXL": ConditioningCropZoomSDXL,
    "ConditioningCutoffSectionsPrompt": ConditioningCutoffSectionsPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningMergeWithTimestepRanges": "Unified Conditioning Merge (with timestep ranges)",
    "DebugConditioning": "Debug Conditioning",
    "CLIPTextEncodeSDXLAutoSplitAndMerge": "CLIPTextEncodeSDXL (auto-split-and-merge)",
    "ConditioningCropZoomSDXL": "Conditioning-crop-zoom-SDXL",
    "ConditioningCutoffSectionsPrompt": "Conditioning Cutoff Sections Prompt",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
