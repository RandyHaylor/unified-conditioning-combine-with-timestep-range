"""
Conditioning-crop-zoom-SDXL

Modifies the SDXL size/crop metadata on every CONDITIONING entry to create a
"free zoom bias" / composition hint. Does NOT actually crop or zoom the
latent — it only rewrites the conditioning metadata that SDXL uses to bias
where the image content sits within an implied source frame.

Behavior:
  target_width / target_height = latent W / H (read from the LATENT input)
  width / height               = round(latent_W * zoom) / round(latent_H * zoom)
                                  (this becomes SDXL's "original size")
  crop_w / crop_h              = derived from offset_x / offset_y in [-1, +1]
                                  -1 = far left / top
                                   0 = centered
                                  +1 = far right / bottom

zoom is clamped to >= 1.0 so the source frame is always at least as large as
the target. Offsets are clamped to [-1.0, +1.0]. Each output CONDITIONING
entry's metadata is updated in place (preserving all other keys like
start_percent, end_percent, pooled_output, strength, etc.).

Both ComfyUI's flat-key form (`width` / `height` / `crop_w` / `crop_h` /
`target_width` / `target_height`) AND the SDXL paper tuple form
(`original_size_as_tuple` / `crop_coords_top_left` / `target_size_as_tuple`)
are written to be compatible with downstream pipelines that read either.
"""

# Latent tensors in ComfyUI are stored at 1/8 resolution of the image space
# (the VAE downscale factor). Multiplying the latent H/W by 8 recovers the
# pixel-space dimensions that the SDXL size-conditioning fields expect.
LATENT_TO_IMAGE_SPATIAL_SCALE_FACTOR = 8

ZOOM_MINIMUM_VALUE = 1.0
ZOOM_MAXIMUM_VALUE = 100.0
ZOOM_DEFAULT_VALUE = 1.0
ZOOM_STEP = 0.01

OFFSET_MINIMUM_VALUE = -1.0
OFFSET_MAXIMUM_VALUE = 1.0
OFFSET_DEFAULT_VALUE = 0.0
OFFSET_STEP = 0.01


def _clamp_numeric_value_inclusive(value_to_clamp, minimum_allowed, maximum_allowed):
    if value_to_clamp < minimum_allowed:
        return minimum_allowed
    if value_to_clamp > maximum_allowed:
        return maximum_allowed
    return value_to_clamp


def _compute_sdxl_size_and_crop_metadata_fields_from_latent_and_zoom_and_offsets(
    latent_width_in_pixels,
    latent_height_in_pixels,
    zoom_factor_clamped_at_or_above_one,
    offset_x_in_negative_one_to_positive_one,
    offset_y_in_negative_one_to_positive_one,
):
    """Returns a dict with all the metadata keys to merge onto each cond entry."""
    target_width = latent_width_in_pixels
    target_height = latent_height_in_pixels

    source_width = int(round(latent_width_in_pixels * zoom_factor_clamped_at_or_above_one))
    source_height = int(round(latent_height_in_pixels * zoom_factor_clamped_at_or_above_one))

    maximum_horizontal_crop_offset = max(0, source_width - target_width)
    maximum_vertical_crop_offset = max(0, source_height - target_height)

    horizontal_offset_normalized_to_zero_one_range = (offset_x_in_negative_one_to_positive_one + 1.0) * 0.5
    vertical_offset_normalized_to_zero_one_range = (offset_y_in_negative_one_to_positive_one + 1.0) * 0.5

    crop_w = int(round(horizontal_offset_normalized_to_zero_one_range * maximum_horizontal_crop_offset))
    crop_h = int(round(vertical_offset_normalized_to_zero_one_range * maximum_vertical_crop_offset))

    return {
        "width": source_width,
        "height": source_height,
        "crop_w": crop_w,
        "crop_h": crop_h,
        "target_width": target_width,
        "target_height": target_height,
        # SDXL paper / diffusers tuple form for pipelines that read these instead.
        "original_size_as_tuple": (source_width, source_height),
        "crop_coords_top_left": (crop_w, crop_h),
        "target_size_as_tuple": (target_width, target_height),
    }


def _apply_metadata_fields_to_each_entry_of_one_conditioning(
    input_conditioning_list_or_none, metadata_fields_to_merge_into_each_entry
):
    """
    Returns a new CONDITIONING list with the same per-entry tensors but each
    entry's metadata dict updated to include the given metadata fields.
    Returns an empty list when the input is None (caller didn't connect this
    optional input).
    """
    if input_conditioning_list_or_none is None:
        return []
    output_conditioning_entries = []
    for conditioning_entry in input_conditioning_list_or_none:
        entry_tokens_tensor = conditioning_entry[0]
        entry_metadata_dict = conditioning_entry[1]
        updated_metadata_dict = dict(entry_metadata_dict)
        updated_metadata_dict.update(metadata_fields_to_merge_into_each_entry)
        output_conditioning_entries.append([entry_tokens_tensor, updated_metadata_dict])
    return output_conditioning_entries


class ConditioningCropZoomSDXL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "zoom": ("FLOAT", {
                    "default": ZOOM_DEFAULT_VALUE,
                    "min": ZOOM_MINIMUM_VALUE,
                    "max": ZOOM_MAXIMUM_VALUE,
                    "step": ZOOM_STEP,
                }),
                "offset_x": ("FLOAT", {
                    "default": OFFSET_DEFAULT_VALUE,
                    "min": OFFSET_MINIMUM_VALUE,
                    "max": OFFSET_MAXIMUM_VALUE,
                    "step": OFFSET_STEP,
                }),
                "offset_y": ("FLOAT", {
                    "default": OFFSET_DEFAULT_VALUE,
                    "min": OFFSET_MINIMUM_VALUE,
                    "max": OFFSET_MAXIMUM_VALUE,
                    "step": OFFSET_STEP,
                }),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_sdxl_zoom_metadata_to_positive_and_negative_conditionings"
    CATEGORY = "unified-conditioning-merge"

    def apply_sdxl_zoom_metadata_to_positive_and_negative_conditionings(
        self, positive, latent, zoom, offset_x, offset_y, negative=None
    ):
        zoom_factor_clamped_at_or_above_one = max(ZOOM_MINIMUM_VALUE, float(zoom))
        offset_x_clamped = _clamp_numeric_value_inclusive(
            float(offset_x), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE
        )
        offset_y_clamped = _clamp_numeric_value_inclusive(
            float(offset_y), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE
        )

        latent_samples_tensor = latent["samples"]
        latent_width_in_pixels = latent_samples_tensor.shape[3] * LATENT_TO_IMAGE_SPATIAL_SCALE_FACTOR
        latent_height_in_pixels = latent_samples_tensor.shape[2] * LATENT_TO_IMAGE_SPATIAL_SCALE_FACTOR

        metadata_fields_to_merge_into_each_entry = (
            _compute_sdxl_size_and_crop_metadata_fields_from_latent_and_zoom_and_offsets(
                latent_width_in_pixels,
                latent_height_in_pixels,
                zoom_factor_clamped_at_or_above_one,
                offset_x_clamped,
                offset_y_clamped,
            )
        )

        output_positive_conditioning_entries = _apply_metadata_fields_to_each_entry_of_one_conditioning(
            positive, metadata_fields_to_merge_into_each_entry
        )
        output_negative_conditioning_entries = _apply_metadata_fields_to_each_entry_of_one_conditioning(
            negative, metadata_fields_to_merge_into_each_entry
        )

        return (output_positive_conditioning_entries, output_negative_conditioning_entries)
