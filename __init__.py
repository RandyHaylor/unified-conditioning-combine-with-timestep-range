from .conditioning_merge_with_timestep_ranges import ConditioningMergeWithTimestepRanges
from .debug_conditioning import DebugConditioning
from .clip_text_encode_sdxl_auto_split_and_merge import CLIPTextEncodeSDXLAutoSplitAndMerge
from .conditioning_crop_zoom_sdxl import ConditioningCropZoomSDXL
from .clip_text_encode_with_cutoff_region_separation import CLIPTextEncodeWithCutoffRegionSeparation

NODE_CLASS_MAPPINGS = {
    "ConditioningMergeWithTimestepRanges": ConditioningMergeWithTimestepRanges,
    "DebugConditioning": DebugConditioning,
    "CLIPTextEncodeSDXLAutoSplitAndMerge": CLIPTextEncodeSDXLAutoSplitAndMerge,
    "ConditioningCropZoomSDXL": ConditioningCropZoomSDXL,
    "CLIPTextEncodeWithCutoffRegionSeparation": CLIPTextEncodeWithCutoffRegionSeparation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ConditioningMergeWithTimestepRanges": "Unified Conditioning Merge (with timestep ranges)",
    "DebugConditioning": "Debug Conditioning",
    "CLIPTextEncodeSDXLAutoSplitAndMerge": "CLIPTextEncodeSDXL (auto-split-and-merge)",
    "ConditioningCropZoomSDXL": "Conditioning-crop-zoom-SDXL",
    "CLIPTextEncodeWithCutoffRegionSeparation": "CLIP Text Encode (Cutoff Region Separation)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
