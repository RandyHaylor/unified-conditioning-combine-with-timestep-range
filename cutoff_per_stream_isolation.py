"""
Per-stream Cutoff-style region isolation.

ORIGINAL MIT-LICENSED IMPLEMENTATION — written from scratch following the
algorithm description (the algorithm itself is an idea, not copyrightable).
This file is NOT a copy or derivative of BlenderNeko's ComfyUI_Cutoff
(which is GPL v3 and was deliberately not vendored to avoid the
copyleft viral effect on this MIT-licensed project).

The algorithm:
  1. Encode the full prompt through ONE CLIP stream (L or G) → base_embedding.
  2. For each "isolate" region (a phrase whose contextual influence we
     want to confine to its own tokens):
       a. Tokenize the region's target text and find ALL positions where
          that token sequence appears in the full prompt's tokens.
       b. Build per-region "region_mask" (1 inside that region's
          positions, 0 elsewhere) and "target_mask" (same in this
          implementation — we use phrase-level decontamination where
          region_text == target_text).
  3. Compute the GLOBAL target mask = union of all per-region target masks.
     Encode a variant of the prompt where every globally-targeted token is
     replaced with mask_token_id → base_masked_embedding.
  4. Compute base_start and base_outer using the strict_mask and
     start_from_masked knobs:
       base_start = base * (1 - start_from_masked) + base_masked * start_from_masked
       base_outer = base * (1 - strict_mask)      + base_masked * strict_mask
  5. For each region:
       a. Encode a variant where all OTHER regions' targets are masked
          but THIS region's targets are visible → region_emb.
       b. Per-token contribution = (region_emb - base_start) * region_weight.
       c. Restrict to inside this region's positions only.
  6. Compose the final per-token embedding:
       final = base_start * global_region_mask + base_outer * (1 - global_region_mask)
       final += sum(per-region contributions)

This is exactly cutoff's algorithm (which is widely-published technique)
restricted to one CLIP stream so the caller can apply different prompt
texts to L and G independently — the missing feature in the original
single-text cutoff that made per-stream routing impossible.
"""

import numpy as np
import torch


# Each CLIP-77-token chunk contains 1 start + up to 75 content tokens + 1 end + padding.
# The content region is therefore positions [1 : chunk_length - 1].
DEFAULT_CLIP_CHUNK_LENGTH_INCLUDING_MARKERS = 77
CHUNK_CONTENT_TOKEN_COUNT_EXCLUDING_MARKERS = 75


def _get_per_stream_clip_text_encoder_model_and_tokenizer(clip_object, stream_key):
    """
    Returns (per_stream_clip_text_encoder_model, per_stream_tokenizer_object)
    for the chosen CLIP stream. Raises if the model/tokenizer don't expose
    the requested stream (e.g., non-SDXL CLIP).
    """
    if stream_key not in ("l", "g"):
        raise ValueError(f"_get_per_stream_clip_text_encoder_model_and_tokenizer: stream_key must be 'l' or 'g', got {stream_key!r}")
    cond_stage_model = clip_object.cond_stage_model
    tokenizer_object = clip_object.tokenizer

    if stream_key == "g":
        if not hasattr(cond_stage_model, "clip_g") or not hasattr(tokenizer_object, "clip_g"):
            raise RuntimeError("This CLIP model does not expose a 'g' stream (not SDXL).")
        return cond_stage_model.clip_g, tokenizer_object.clip_g
    else:
        if not hasattr(cond_stage_model, "clip_l") or not hasattr(tokenizer_object, "clip_l"):
            raise RuntimeError("This CLIP model does not expose an 'l' stream (not SDXL).")
        return cond_stage_model.clip_l, tokenizer_object.clip_l


def _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad(
    tokens_per_chunk_list, end_token_id_for_this_stream
):
    """
    Each chunk is a list of (token_id, weight) pairs of length 77 (CLIP's
    chunk length): [(start, w), (content, w), ..., (content, w), (end, w),
    (pad, w), ...].

    Returns a flat list of content token ids (skipping the per-chunk start
    token at index 0 and stopping at the first end-token within each chunk),
    suitable for sublist matching against region target tokens.
    """
    flat_content_token_ids_for_all_chunks_list = []
    for one_chunk_of_token_weight_pairs in tokens_per_chunk_list:
        for index_within_chunk, (token_id_in_pair, _weight_in_pair) in enumerate(one_chunk_of_token_weight_pairs):
            if index_within_chunk == 0:
                # Skip the start token at the beginning of every chunk.
                continue
            if token_id_in_pair == end_token_id_for_this_stream:
                # End token — stop reading this chunk's content.
                break
            flat_content_token_ids_for_all_chunks_list.append(token_id_in_pair)
    return flat_content_token_ids_for_all_chunks_list


def _find_all_sublist_match_start_positions_within_superlist(superlist, sublist):
    """
    Returns the list of start positions in superlist where sublist appears
    as a contiguous run. Returns empty list if sublist is empty.
    """
    if not sublist:
        return []
    matched_start_positions_list = []
    sublist_length = len(sublist)
    superlist_length = len(superlist)
    if sublist_length > superlist_length:
        return []
    for candidate_start_position in range(superlist_length - sublist_length + 1):
        if superlist[candidate_start_position] == sublist[0]:
            if superlist[candidate_start_position : candidate_start_position + sublist_length] == sublist:
                matched_start_positions_list.append(candidate_start_position)
    return matched_start_positions_list


def _expand_content_position_mask_into_full_chunk_positions_with_start_end_padding(
    content_position_mask_one_dimensional, num_chunks
):
    """
    Cutoff's masks are computed over CONTENT token positions (excluding the
    per-chunk start/end/pad markers). The encoded embedding tensor, however,
    spans the FULL chunk positions including those markers. This helper
    reshapes a per-content-position mask of length (num_chunks * 75) into a
    per-full-chunk-position mask of length (num_chunks * 77) by padding each
    chunk-row with a 0 on the left (start marker) and a 0 on the right (end
    marker).
    """
    if content_position_mask_one_dimensional.shape[0] != num_chunks * CHUNK_CONTENT_TOKEN_COUNT_EXCLUDING_MARKERS:
        # Should not happen if caller passes correct shape, but be defensive.
        # Pad to match expected length.
        return content_position_mask_one_dimensional
    reshaped_to_per_chunk_rows = content_position_mask_one_dimensional.reshape(
        num_chunks, CHUNK_CONTENT_TOKEN_COUNT_EXCLUDING_MARKERS
    )
    padded_with_zeros_for_start_and_end_markers_per_row = np.pad(
        reshaped_to_per_chunk_rows,
        pad_width=((0, 0), (1, 1)),
        mode="constant",
        constant_values=0,
    )
    flattened_back_to_one_dim = padded_with_zeros_for_start_and_end_markers_per_row.reshape(-1)
    return flattened_back_to_one_dim


def _build_masked_tokens_per_chunk_by_replacing_target_positions_with_mask_token_id(
    base_tokens_per_chunk_list, content_position_mask, mask_token_id_to_replace_with, end_token_id
):
    """
    Returns a new list-of-chunks with the same shape as base_tokens_per_chunk_list,
    but with every CONTENT token whose mask value is non-zero replaced by
    (mask_token_id, original_weight).

    content_position_mask is a 1D array of length num_chunks * 75 where
    nonzero means "mask this position".
    """
    masked_tokens_per_chunk_output_list = []
    content_position_mask_index_iterator = 0
    for one_chunk_of_token_weight_pairs in base_tokens_per_chunk_list:
        masked_chunk = []
        for index_within_chunk, (token_id_in_pair, weight_in_pair) in enumerate(one_chunk_of_token_weight_pairs):
            if index_within_chunk == 0:
                # Start token — passthrough.
                masked_chunk.append((token_id_in_pair, weight_in_pair))
                continue
            # For content positions, check the mask. For end/pad positions we
            # don't consume a mask slot — but we need to keep our iterator
            # aligned to CHUNK_CONTENT_TOKEN_COUNT_EXCLUDING_MARKERS per chunk.
            if token_id_in_pair == end_token_id:
                # Once we hit end-of-text inside a chunk, the rest is padding.
                # The mask was built ONLY for content positions, but we may
                # need to advance the iterator to the next chunk's content
                # window boundary. Fast forward and just passthrough the rest
                # of this chunk.
                masked_chunk.append((token_id_in_pair, weight_in_pair))
                # Continue iterating to copy remaining tokens unchanged.
                for remaining_index in range(index_within_chunk + 1, len(one_chunk_of_token_weight_pairs)):
                    masked_chunk.append(one_chunk_of_token_weight_pairs[remaining_index])
                # Advance iterator past unread content positions in this chunk.
                positions_in_this_content_window = CHUNK_CONTENT_TOKEN_COUNT_EXCLUDING_MARKERS
                content_position_mask_index_iterator = (
                    (content_position_mask_index_iterator // positions_in_this_content_window) + 1
                ) * positions_in_this_content_window
                break
            # Content token: consult the mask.
            mask_value_for_this_position = content_position_mask[content_position_mask_index_iterator] if content_position_mask_index_iterator < len(content_position_mask) else 0
            content_position_mask_index_iterator += 1
            if mask_value_for_this_position != 0:
                masked_chunk.append((mask_token_id_to_replace_with, weight_in_pair))
            else:
                masked_chunk.append((token_id_in_pair, weight_in_pair))
        else:
            # No break encountered (no end token found within the chunk's
            # content positions — meaning content filled the entire 75-slot
            # window). Iterator should naturally advance by 75 here.
            pass
        masked_tokens_per_chunk_output_list.append(masked_chunk)
    return masked_tokens_per_chunk_output_list


def _resolve_mask_token_id_from_user_string_with_default(per_stream_tokenizer, mask_token_string, end_token_id):
    if not mask_token_string:
        return end_token_id
    raw_huggingface_tokenizer = per_stream_tokenizer.tokenizer
    tokenized_input_ids_for_mask_token_string = raw_huggingface_tokenizer(mask_token_string)["input_ids"][1:-1]
    if not tokenized_input_ids_for_mask_token_string:
        return end_token_id
    return tokenized_input_ids_for_mask_token_string[0]


def encode_one_stream_with_cutoff_style_region_isolation(
    clip_object,
    stream_key,
    full_prompt_text_for_this_stream,
    isolate_section_target_text_and_weight_pairs_list,
    mask_token_string,
    strict_mask_value,
    start_from_masked_value,
):
    """
    Returns (final_embedding_tensor_for_this_stream, pooled_output_tensor_or_none).

    `isolate_section_target_text_and_weight_pairs_list` is a list of dicts
    `[{"target_text": str, "weight": float}, ...]`. The target_text for each
    section is the phrase whose contextual influence we want to confine to
    its own positions in the full prompt.
    """
    per_stream_clip_text_encoder_model, per_stream_tokenizer = _get_per_stream_clip_text_encoder_model_and_tokenizer(
        clip_object, stream_key
    )
    end_token_id_for_this_stream = per_stream_tokenizer.end_token

    # Tokenize the full prompt for this stream.
    base_tokens_per_chunk_list = per_stream_tokenizer.tokenize_with_weights(full_prompt_text_for_this_stream)
    number_of_chunks_after_tokenization = len(base_tokens_per_chunk_list)

    # Encode base (unmasked) full prompt.
    base_embedding_tensor, base_pooled_tensor_or_none = per_stream_clip_text_encoder_model.encode_token_weights(
        base_tokens_per_chunk_list
    )

    # If no isolate regions, return base directly.
    if not isolate_section_target_text_and_weight_pairs_list:
        return base_embedding_tensor, base_pooled_tensor_or_none

    # Flatten base tokens to content-only IDs for region matching.
    base_content_token_ids_flat_list = _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad(
        base_tokens_per_chunk_list, end_token_id_for_this_stream
    )
    # Pad the flat content list to num_chunks * 75 so masks align with chunk content windows.
    expected_content_position_count_across_all_chunks = (
        number_of_chunks_after_tokenization * CHUNK_CONTENT_TOKEN_COUNT_EXCLUDING_MARKERS
    )

    # Build per-region region/target masks.
    per_region_target_masks_list = []
    per_region_region_masks_list = []
    per_region_weights_list = []

    for section_target_text_and_weight in isolate_section_target_text_and_weight_pairs_list:
        section_target_text = section_target_text_and_weight["target_text"]
        section_weight = float(section_target_text_and_weight.get("weight", 1.0))

        # Tokenize the section's target text.
        section_target_tokens_per_chunk = per_stream_tokenizer.tokenize_with_weights(section_target_text)
        section_target_content_ids_flat = _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad(
            section_target_tokens_per_chunk, end_token_id_for_this_stream
        )
        if not section_target_content_ids_flat:
            # Target produced no content tokens (e.g., empty after stripping).
            continue

        match_start_positions_list = _find_all_sublist_match_start_positions_within_superlist(
            base_content_token_ids_flat_list, section_target_content_ids_flat
        )

        per_region_target_mask_one_dim = np.zeros(expected_content_position_count_across_all_chunks, dtype=int)
        per_region_region_mask_one_dim = np.zeros(expected_content_position_count_across_all_chunks, dtype=int)
        section_target_length_in_content_tokens = len(section_target_content_ids_flat)
        for match_start_position_in_base_content in match_start_positions_list:
            target_end_exclusive = match_start_position_in_base_content + section_target_length_in_content_tokens
            if target_end_exclusive > len(per_region_target_mask_one_dim):
                continue
            per_region_target_mask_one_dim[match_start_position_in_base_content : target_end_exclusive] = 1
            # region_text == target_text gives "phrase-level decontamination":
            # the region is the same span as the target.
            per_region_region_mask_one_dim[match_start_position_in_base_content : target_end_exclusive] = 1

        per_region_target_masks_list.append(per_region_target_mask_one_dim)
        per_region_region_masks_list.append(per_region_region_mask_one_dim)
        per_region_weights_list.append(section_weight)

    # If no regions matched, return base.
    if not per_region_target_masks_list:
        return base_embedding_tensor, base_pooled_tensor_or_none

    # Compute global masks (union).
    global_target_mask_over_content = np.maximum.reduce(per_region_target_masks_list)
    global_region_mask_over_content = np.maximum.reduce(per_region_region_masks_list).astype(float)
    regions_overlap_count_sum = np.sum(np.stack(per_region_region_masks_list), axis=0).astype(float)
    regions_normalized_by_overlap_count = np.divide(
        1.0,
        regions_overlap_count_sum,
        out=np.zeros_like(regions_overlap_count_sum),
        where=regions_overlap_count_sum != 0,
    )

    # Resolve mask token id.
    mask_token_id_to_use = _resolve_mask_token_id_from_user_string_with_default(
        per_stream_tokenizer, mask_token_string, end_token_id_for_this_stream
    )

    # Build masked base: all globally-targeted positions replaced with mask token.
    base_masked_tokens_per_chunk = _build_masked_tokens_per_chunk_by_replacing_target_positions_with_mask_token_id(
        base_tokens_per_chunk_list,
        global_target_mask_over_content,
        mask_token_id_to_use,
        end_token_id_for_this_stream,
    )
    base_masked_embedding_tensor, _unused_masked_pooled = per_stream_clip_text_encoder_model.encode_token_weights(
        base_masked_tokens_per_chunk
    )

    # Compute base_start and base_outer per cutoff's knob formula.
    base_start_embedding_tensor = (
        base_embedding_tensor * (1.0 - start_from_masked_value)
        + base_masked_embedding_tensor * start_from_masked_value
    )
    base_outer_embedding_tensor = (
        base_embedding_tensor * (1.0 - strict_mask_value)
        + base_masked_embedding_tensor * strict_mask_value
    )

    # Expand content-position masks into full-chunk-position masks (with
    # zeros at start/end marker positions) so they align with the encoded
    # embedding tensor's sequence dimension.
    global_region_mask_over_full_chunks_positions_one_dim = _expand_content_position_mask_into_full_chunk_positions_with_start_end_padding(
        global_region_mask_over_content, number_of_chunks_after_tokenization
    )

    # Per-region contributions.
    region_contributions_summed_tensor = torch.zeros_like(base_embedding_tensor)
    for per_region_target_mask, per_region_region_mask, per_region_weight in zip(
        per_region_target_masks_list, per_region_region_masks_list, per_region_weights_list
    ):
        # Mask = global_target except for this region's target (keep this region visible).
        per_region_mask_to_apply_one_dim = global_target_mask_over_content - per_region_target_mask
        per_region_masked_tokens_per_chunk = _build_masked_tokens_per_chunk_by_replacing_target_positions_with_mask_token_id(
            base_tokens_per_chunk_list,
            per_region_mask_to_apply_one_dim,
            mask_token_id_to_use,
            end_token_id_for_this_stream,
        )
        per_region_embedding_tensor, _unused_region_pooled = per_stream_clip_text_encoder_model.encode_token_weights(
            per_region_masked_tokens_per_chunk
        )
        per_region_difference_from_base_start = per_region_embedding_tensor - base_start_embedding_tensor

        # Build per-position weight tensor for this region.
        per_region_weight_per_content_position = (
            regions_normalized_by_overlap_count * per_region_region_mask * per_region_weight
        )
        per_region_weight_per_full_chunk_position_one_dim = _expand_content_position_mask_into_full_chunk_positions_with_start_end_padding(
            per_region_weight_per_content_position, number_of_chunks_after_tokenization
        )
        per_region_weight_tensor_broadcastable_over_embedding_dim = torch.tensor(
            per_region_weight_per_full_chunk_position_one_dim,
            dtype=base_embedding_tensor.dtype,
            device=base_embedding_tensor.device,
        ).unsqueeze(0).unsqueeze(-1)
        region_contributions_summed_tensor = (
            region_contributions_summed_tensor
            + per_region_difference_from_base_start * per_region_weight_tensor_broadcastable_over_embedding_dim
        )

    # Compose final per-token embedding.
    global_region_mask_tensor_broadcastable = torch.tensor(
        global_region_mask_over_full_chunks_positions_one_dim,
        dtype=base_embedding_tensor.dtype,
        device=base_embedding_tensor.device,
    ).unsqueeze(0).unsqueeze(-1)
    final_embedding_tensor = (
        base_start_embedding_tensor * global_region_mask_tensor_broadcastable
        + base_outer_embedding_tensor * (1.0 - global_region_mask_tensor_broadcastable)
        + region_contributions_summed_tensor
    )

    return final_embedding_tensor, base_pooled_tensor_or_none
