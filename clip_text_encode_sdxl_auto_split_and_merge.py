"""
CLIPTextEncodeSDXL (auto-split-and-merge)

Drop-in extension of stock CLIPTextEncodeSDXL with two dropdowns that control
what happens when a prompt is long enough that the CLIP tokenizer splits it
into more than one 77-token chunk:

    split_and_merge_g   ∈ {truncate, combine, average}     (CLIP-G stream)
    split_and_merge_l   ∈ {truncate, combine, average}     (CLIP-L stream)

Per-mode behavior:
    truncate   — keep only the FIRST 77-token chunk of that stream
                 (any overflow tokens are dropped; default, matches
                 most users' implicit expectation of stock encoders).
    combine    — keep ALL chunks of that stream as separate parallel
                 branches in the output CONDITIONING list.
    average    — keep all chunks, encode each, then blend their
                 resulting embeddings (and pooled_output) into a
                 single 77-token cond entry.

The two dropdowns are independent. The final output count depends on the
COMBINATION:
    - If either stream is `combine`  → multi-entry output (one per chunk pair).
    - Else if either stream is `average` → single-entry output (averaged).
    - Else (both truncate)             → single-entry output (one chunk).

Output:
    conditioning  (CONDITIONING) — usable in any sampler; multi-entry when
                                   `combine` is selected on either stream.
    debug_info    (STRING)       — description of how many chunks each
                                   stream had and how the result was built.

Stock CLIPTextEncodeSDXL behavior is reproduced when both dropdowns are
left at `truncate`.

Source reference: ComfyUI/comfy_extras/nodes_clip_sdxl.py:28 (CLIPTextEncodeSDXL).
"""

import torch

import nodes


SPLIT_AND_MERGE_MODE_TRUNCATE = "truncate"
SPLIT_AND_MERGE_MODE_COMBINE = "combine"
SPLIT_AND_MERGE_MODE_AVERAGE = "average"

SPLIT_AND_MERGE_MODE_CHOICES_IN_DROPDOWN_ORDER = [
    SPLIT_AND_MERGE_MODE_TRUNCATE,
    SPLIT_AND_MERGE_MODE_COMBINE,
    SPLIT_AND_MERGE_MODE_AVERAGE,
]


def _reduce_chunks_list_to_a_single_first_chunk_when_truncating(
    tokenizer_chunks_list, split_and_merge_mode
):
    if split_and_merge_mode == SPLIT_AND_MERGE_MODE_TRUNCATE and len(tokenizer_chunks_list) > 0:
        return tokenizer_chunks_list[:1]
    return list(tokenizer_chunks_list)


def _pad_two_chunk_lists_to_matching_length_with_empty_chunks(
    chunks_list_for_g_stream, chunks_list_for_l_stream, clip
):
    target_chunk_count = max(len(chunks_list_for_g_stream), len(chunks_list_for_l_stream))
    empty_tokens_dict = clip.tokenize("")
    while len(chunks_list_for_g_stream) < target_chunk_count:
        chunks_list_for_g_stream.append(empty_tokens_dict["g"][0])
    while len(chunks_list_for_l_stream) < target_chunk_count:
        chunks_list_for_l_stream.append(empty_tokens_dict["l"][0])
    return chunks_list_for_g_stream, chunks_list_for_l_stream


def _average_per_chunk_conditionings_into_a_single_conditioning_entry(per_chunk_conditioning_list):
    if len(per_chunk_conditioning_list) == 0:
        raise ValueError("CLIPTextEncodeSDXLAutoSplitAndMerge: cannot average an empty list of per-chunk conditionings.")

    first_conditioning = per_chunk_conditioning_list[0]
    first_entry_tokens_tensor = first_conditioning[0][0]
    first_entry_metadata_dict = first_conditioning[0][1]

    summed_tokens_tensor = torch.zeros_like(first_entry_tokens_tensor)
    summed_pooled_output_tensor = None
    pooled_contribution_count = 0
    total_chunk_count = 0

    for per_chunk_conditioning in per_chunk_conditioning_list:
        entry_tokens_tensor = per_chunk_conditioning[0][0]
        entry_metadata_dict = per_chunk_conditioning[0][1]
        summed_tokens_tensor = summed_tokens_tensor + entry_tokens_tensor
        total_chunk_count += 1
        entry_pooled_output = entry_metadata_dict.get("pooled_output", None)
        if entry_pooled_output is not None:
            if summed_pooled_output_tensor is None:
                summed_pooled_output_tensor = torch.zeros_like(entry_pooled_output)
            summed_pooled_output_tensor = summed_pooled_output_tensor + entry_pooled_output
            pooled_contribution_count += 1

    averaged_tokens_tensor = summed_tokens_tensor / float(total_chunk_count)
    averaged_metadata_dict = first_entry_metadata_dict.copy()
    if summed_pooled_output_tensor is not None and pooled_contribution_count > 0:
        averaged_metadata_dict["pooled_output"] = summed_pooled_output_tensor / float(pooled_contribution_count)

    return [averaged_tokens_tensor, averaged_metadata_dict]


class CLIPTextEncodeSDXLAutoSplitAndMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION}),
                "height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION}),
                "target_width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION}),
                "target_height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION}),
                "text_g": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "text_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "split_and_merge_g": (
                    SPLIT_AND_MERGE_MODE_CHOICES_IN_DROPDOWN_ORDER,
                    {"default": SPLIT_AND_MERGE_MODE_TRUNCATE},
                ),
                "split_and_merge_l": (
                    SPLIT_AND_MERGE_MODE_CHOICES_IN_DROPDOWN_ORDER,
                    {"default": SPLIT_AND_MERGE_MODE_TRUNCATE},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "debug_info")
    FUNCTION = "encode_with_auto_split_and_merge"
    CATEGORY = "unified-conditioning-merge"

    def encode_with_auto_split_and_merge(
        self,
        clip,
        width,
        height,
        crop_w,
        crop_h,
        target_width,
        target_height,
        text_g,
        text_l,
        split_and_merge_g,
        split_and_merge_l,
    ):
        raw_tokenizer_output_for_g_stream = clip.tokenize(text_g)
        raw_tokenizer_output_for_l_stream = clip.tokenize(text_l)

        all_chunks_from_g_stream_before_reduction = list(raw_tokenizer_output_for_g_stream["g"])
        all_chunks_from_l_stream_before_reduction = list(raw_tokenizer_output_for_l_stream["l"])
        initial_chunk_count_g = len(all_chunks_from_g_stream_before_reduction)
        initial_chunk_count_l = len(all_chunks_from_l_stream_before_reduction)

        chunks_to_encode_for_g_stream = _reduce_chunks_list_to_a_single_first_chunk_when_truncating(
            all_chunks_from_g_stream_before_reduction, split_and_merge_g
        )
        chunks_to_encode_for_l_stream = _reduce_chunks_list_to_a_single_first_chunk_when_truncating(
            all_chunks_from_l_stream_before_reduction, split_and_merge_l
        )

        chunks_to_encode_for_g_stream, chunks_to_encode_for_l_stream = (
            _pad_two_chunk_lists_to_matching_length_with_empty_chunks(
                chunks_to_encode_for_g_stream, chunks_to_encode_for_l_stream, clip
            )
        )

        sdxl_size_conditioning_add_dict = {
            "width": width,
            "height": height,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "target_width": target_width,
            "target_height": target_height,
        }

        total_paired_chunk_count_to_encode = len(chunks_to_encode_for_g_stream)
        per_paired_chunk_conditioning_list = []
        for paired_chunk_index in range(total_paired_chunk_count_to_encode):
            tokens_for_this_pair = {
                "g": [chunks_to_encode_for_g_stream[paired_chunk_index]],
                "l": [chunks_to_encode_for_l_stream[paired_chunk_index]],
            }
            encoded_conditioning_for_this_pair = clip.encode_from_tokens_scheduled(
                tokens_for_this_pair, add_dict=sdxl_size_conditioning_add_dict
            )
            per_paired_chunk_conditioning_list.append(encoded_conditioning_for_this_pair)

        any_stream_is_combine_mode = (
            split_and_merge_g == SPLIT_AND_MERGE_MODE_COMBINE
            or split_and_merge_l == SPLIT_AND_MERGE_MODE_COMBINE
        )
        any_stream_is_average_mode = (
            split_and_merge_g == SPLIT_AND_MERGE_MODE_AVERAGE
            or split_and_merge_l == SPLIT_AND_MERGE_MODE_AVERAGE
        )

        if any_stream_is_combine_mode:
            output_conditioning_list = []
            for per_pair_conditioning in per_paired_chunk_conditioning_list:
                output_conditioning_list.extend(per_pair_conditioning)
            output_summary_text = "combine"
        elif any_stream_is_average_mode:
            averaged_single_entry = _average_per_chunk_conditionings_into_a_single_conditioning_entry(
                per_paired_chunk_conditioning_list
            )
            output_conditioning_list = [averaged_single_entry]
            output_summary_text = "average"
        else:
            output_conditioning_list = per_paired_chunk_conditioning_list[0]
            output_summary_text = "truncate (first chunk only)"

        debug_info_string = (
            f"CLIPTextEncodeSDXL (auto-split-and-merge):\n"
            f"  text_g chunks produced by tokenizer: {initial_chunk_count_g}\n"
            f"  text_l chunks produced by tokenizer: {initial_chunk_count_l}\n"
            f"  split_and_merge_g: {split_and_merge_g}\n"
            f"  split_and_merge_l: {split_and_merge_l}\n"
            f"  paired chunk count actually encoded: {total_paired_chunk_count_to_encode}\n"
            f"  output reduction strategy: {output_summary_text}\n"
            f"  final CONDITIONING entry count: {len(output_conditioning_list)}"
        )

        return (output_conditioning_list, debug_info_string)
