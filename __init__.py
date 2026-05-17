from .conditioning_merge_with_timestep_ranges import ConditioningMergeWithTimestepRanges
from .debug_conditioning import DebugConditioning
from .clip_text_encode_sdxl_auto_split_and_merge import CLIPTextEncodeSDXLAutoSplitAndMerge
from .conditioning_crop_zoom_sdxl import ConditioningCropZoomSDXL
from .clip_text_encode_with_cutoff_region_separation import CLIPTextEncodeWithCutoffRegionSeparation

# Side-effect import: registers HTTP routes used by the cutoff node's
# realtime embedding-validation widget. Failure here should not block
# plugin loading — the validator just won't function if routes don't
# register (e.g., older ComfyUI without PromptServer API).
try:
    from . import server_routes  # noqa: F401
except Exception as _server_routes_import_failure:
    import logging as _logging_for_server_routes_failure_warning
    _logging_for_server_routes_failure_warning.warning(
        f"unified-conditioning-merge: server routes failed to register "
        f"({type(_server_routes_import_failure).__name__}: {_server_routes_import_failure}). "
        f"Realtime embedding validation in the cutoff node will be unavailable."
    )

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
    "CLIPTextEncodeWithCutoffRegionSeparation": "CLIP Text Encode SDXL (Cutoff Region Enhanced)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
