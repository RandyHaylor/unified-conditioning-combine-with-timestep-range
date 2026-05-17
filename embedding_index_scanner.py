"""
Lazy, in-memory index of every textual-inversion embedding file under
ComfyUI's embeddings directory and the tensor last-dim(s) each file
contains. Used by the realtime validation endpoint to classify a prompt's
`embedding:NAME` references as "not found", "incompatible with SDXL", or
OK without ever loading a CLIP model.

The index is built lazily on first call and cached process-wide. Call
`invalidate_cached_embedding_filename_to_dim_index()` to force a rescan
(used by the rescan HTTP route).

Per-file dim detection:
  - `.safetensors` files: read tensor shapes via the file's header only
    (no actual tensor data is loaded into memory).
  - `.pt` / `.bin` / `.ckpt` files: load with `torch.load(weights_only=True,
    map_location='cpu')` and walk the resulting object recursively.

For SDXL compatibility classification, an embedding file is considered
compatible only when its set of tensor last-dims contains BOTH 768 (the
CLIP-L stream's expected dim) AND 1280 (the CLIP-G stream's expected dim).
Pure-SD1.5 embeddings have only 768; they are flagged incompatible because
the SDXL G stream would silently drop them at encode time.
"""

import logging
import os


_cached_embedding_lowercase_stem_to_index_entry_map = None


SDXL_CLIP_L_EXPECTED_DIM = 768
SDXL_CLIP_G_EXPECTED_DIM = 1280
SDXL_REQUIRED_TENSOR_DIM_SET_THAT_AN_EMBEDDING_FILE_MUST_CONTAIN_TO_BE_FULLY_COMPATIBLE = frozenset(
    {SDXL_CLIP_L_EXPECTED_DIM, SDXL_CLIP_G_EXPECTED_DIM}
)


def invalidate_cached_embedding_filename_to_dim_index():
    global _cached_embedding_lowercase_stem_to_index_entry_map
    _cached_embedding_lowercase_stem_to_index_entry_map = None


def _read_tensor_last_dim_set_from_safetensors_file_header_only(file_path):
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    try:
        dims_seen = set()
        with safe_open(file_path, framework="pt") as opened_safetensors_file_handle:
            for tensor_key_in_file in opened_safetensors_file_handle.keys():
                tensor_shape_tuple_or_list = opened_safetensors_file_handle.get_slice(
                    tensor_key_in_file
                ).get_shape()
                if tensor_shape_tuple_or_list and len(tensor_shape_tuple_or_list) > 0:
                    dims_seen.add(int(tensor_shape_tuple_or_list[-1]))
        return frozenset(dims_seen)
    except Exception:
        return None


def _read_tensor_last_dim_set_from_pt_or_bin_file_via_full_load(file_path):
    try:
        import torch
    except ImportError:
        return None
    try:
        loaded_payload_from_pickle_file = torch.load(
            file_path, weights_only=True, map_location="cpu"
        )
    except Exception:
        return None
    return _recursively_gather_tensor_last_dim_set_from_arbitrary_payload(loaded_payload_from_pickle_file)


def _recursively_gather_tensor_last_dim_set_from_arbitrary_payload(payload):
    try:
        import torch
    except ImportError:
        return frozenset()
    accumulated_dims_set = set()
    if isinstance(payload, torch.Tensor):
        if payload.dim() > 0:
            accumulated_dims_set.add(int(payload.shape[-1]))
    elif isinstance(payload, dict):
        for value_in_dict in payload.values():
            accumulated_dims_set |= _recursively_gather_tensor_last_dim_set_from_arbitrary_payload(
                value_in_dict
            )
    elif isinstance(payload, (list, tuple)):
        for item_in_sequence in payload:
            accumulated_dims_set |= _recursively_gather_tensor_last_dim_set_from_arbitrary_payload(
                item_in_sequence
            )
    return frozenset(accumulated_dims_set)


def _build_full_embedding_lowercase_stem_to_index_entry_map_from_disk_scan():
    try:
        import folder_paths
    except ImportError:
        return {}
    try:
        all_embedding_filenames_in_directory = folder_paths.get_filename_list("embeddings")
    except Exception:
        return {}

    new_index_under_construction = {}
    for one_embedding_filename in all_embedding_filenames_in_directory:
        try:
            absolute_path_to_this_embedding_file = folder_paths.get_full_path(
                "embeddings", one_embedding_filename
            )
        except Exception:
            continue
        if not absolute_path_to_this_embedding_file or not os.path.isfile(
            absolute_path_to_this_embedding_file
        ):
            continue

        file_extension_lowercase_with_dot = os.path.splitext(one_embedding_filename)[1].lower()
        if file_extension_lowercase_with_dot == ".safetensors":
            tensor_last_dim_set_for_this_file = _read_tensor_last_dim_set_from_safetensors_file_header_only(
                absolute_path_to_this_embedding_file
            )
        else:
            tensor_last_dim_set_for_this_file = _read_tensor_last_dim_set_from_pt_or_bin_file_via_full_load(
                absolute_path_to_this_embedding_file
            )

        if tensor_last_dim_set_for_this_file is None:
            tensor_last_dim_set_for_this_file = frozenset()

        original_stem_without_extension = os.path.splitext(one_embedding_filename)[0]
        new_index_under_construction[original_stem_without_extension.lower()] = {
            "original_filename": one_embedding_filename,
            "original_stem": original_stem_without_extension,
            "tensor_last_dim_set": tensor_last_dim_set_for_this_file,
        }

    logging.info(
        f"unified-conditioning-merge: scanned embeddings directory, "
        f"indexed {len(new_index_under_construction)} embedding file(s)."
    )
    return new_index_under_construction


def get_cached_or_build_embedding_lowercase_stem_to_index_entry_map():
    global _cached_embedding_lowercase_stem_to_index_entry_map
    if _cached_embedding_lowercase_stem_to_index_entry_map is None:
        _cached_embedding_lowercase_stem_to_index_entry_map = (
            _build_full_embedding_lowercase_stem_to_index_entry_map_from_disk_scan()
        )
    return _cached_embedding_lowercase_stem_to_index_entry_map


def is_embedding_file_fully_compatible_with_sdxl_based_on_its_tensor_last_dim_set(
    tensor_last_dim_set_for_one_embedding_file,
):
    return SDXL_REQUIRED_TENSOR_DIM_SET_THAT_AN_EMBEDDING_FILE_MUST_CONTAIN_TO_BE_FULLY_COMPATIBLE.issubset(
        tensor_last_dim_set_for_one_embedding_file
    )
