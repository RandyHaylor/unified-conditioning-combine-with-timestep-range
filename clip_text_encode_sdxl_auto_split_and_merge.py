"""
CLIPTextEncodeSDXL (auto-split-and-merge)

Drop-in extension of stock CLIPTextEncodeSDXL with two dropdowns that control
what happens when a prompt exceeds 77 tokens AND with smart pre-tokenize
splitting at BREAK / comma / line / space boundaries so chunk sizes stay
balanced (no tiny over-weighted tail chunks).

    split_and_merge_g   ∈ {concat, truncate, combine, average}    (CLIP-G)
    split_and_merge_l   ∈ {concat, truncate, combine, average}    (CLIP-L)

Default for both is `concat`. The pipeline that runs is determined by the
highest-priority mode across both streams, in this order (highest first):

    combine > average > concat > truncate

So if either stream picks combine, the combine pipeline runs (multi-entry
output). If either picks average, the average pipeline runs (single
blended entry). If both pick concat, the concat pipeline runs (stock-style
single encode call with all chunks, seq-dim concat). If both pick truncate,
only first chunks are encoded.

Split markers, highest priority first:
    BREAK         — capitalized whole word; same convention as A1111/Forge.
                    Acts as a HARD boundary — sub-balancing never crosses it.
    comma         — `,`
    line break    — `\\n`
    whitespace    — falls back to any whitespace
    character     — last-resort hard split (preserves nothing; should be rare).

Balancing strategy (when one BREAK-segment exceeds 75 content tokens):
    Compute K = ceil(segment_tokens / 75). Sub-split the segment at the
    highest-priority delimiter whose pieces each fit, then greedy-bin-pack
    pieces into K bins targeting an even per-bin token count. Worst-case
    fallback: equal character-length slicing.

Per-mode behavior after splitting:
    truncate   — keep only the FIRST text piece of that stream.
    combine    — keep ALL pieces; each becomes a separate CONDITIONING entry
                 (parallel branches).
    average    — keep all pieces, encode each, then blend the resulting
                 embeddings into a single CONDITIONING entry.

The two dropdowns are independent. Final output count depends on the
COMBINATION:
    - If either stream is `combine`  → multi-entry output.
    - Else if either is `average`    → single-entry averaged output.
    - Else (both truncate)           → single-entry output (one chunk).

Outputs:
    conditioning  (CONDITIONING) — usable in any sampler; multi-entry when
                                   `combine` is selected on either stream.
    debug_info    (STRING)       — description of how splitting and merging
                                   happened, including piece counts per
                                   stream and per-piece token counts.

Reference: comfy_extras/nodes_clip_sdxl.py:28 (stock CLIPTextEncodeSDXL).
"""

import math
import re

import torch

import nodes


SPLIT_AND_MERGE_MODE_CONCAT = "concat (default)"
SPLIT_AND_MERGE_MODE_TRUNCATE = "truncate"
SPLIT_AND_MERGE_MODE_COMBINE = "combine"
SPLIT_AND_MERGE_MODE_AVERAGE = "average"

# Dropdown order. `concat (default)` is first → default. It reproduces
# stock CLIPTextEncodeSDXL's "encode every chunk in one call, concat the
# resulting 77-token outputs along the sequence dim" behavior, while still
# respecting BREAK markers as forced chunk boundaries. `truncate` is LAST
# because it's the least-commonly-wanted option (it drops overflow tokens).
SPLIT_AND_MERGE_MODE_CHOICES_IN_DROPDOWN_ORDER = [
    SPLIT_AND_MERGE_MODE_CONCAT,
    SPLIT_AND_MERGE_MODE_COMBINE,
    SPLIT_AND_MERGE_MODE_AVERAGE,
    SPLIT_AND_MERGE_MODE_TRUNCATE,
]

# Highest-priority mode determines which encoding pipeline the node runs.
# combine > average > concat > truncate. When both streams are concat, we
# use the stock-style pipeline (one encode call, multi-chunk tokens, one
# cond out). When either is combine or average, we use the per-piece
# pipeline (one encode call per piece pair, then either keep multiple
# entries or average them into one).
SPLIT_AND_MERGE_MODE_PIPELINE_PRIORITY_ORDER_HIGHEST_FIRST = [
    SPLIT_AND_MERGE_MODE_COMBINE,
    SPLIT_AND_MERGE_MODE_AVERAGE,
    SPLIT_AND_MERGE_MODE_CONCAT,
    SPLIT_AND_MERGE_MODE_TRUNCATE,
]

# Each CLIP-77-token chunk has start + end + content + padding. Real content
# tokens per chunk: 77 - 2 (start/end) = 75.
CLIP_TOKENS_PER_CHUNK_INCLUDING_MARKERS = 77
CLIP_CONTENT_TOKENS_PER_CHUNK_EXCLUDING_MARKERS = 75

# BREAK is matched as the standalone uppercase word, not as a substring of
# something else. \b would also match e.g. "BREAKDOWN" so use no-non-space
# lookarounds.
BREAK_MARKER_REGEX_PATTERN = re.compile(r"(?:(?<=\s)|(?<=^))BREAK(?=\s|$)")

# Delimiter cascade for sub-splitting a BREAK segment that's still too long.
# Each tuple is (split_regex_pattern, rejoin_string).
DELIMITER_CASCADE_FOR_SUBSPLITTING_OVERLONG_SEGMENTS = [
    (re.compile(r"\s*,\s*"), ", "),     # comma
    (re.compile(r"\s*\n\s*"), "\n"),   # line break
    (re.compile(r"\s+"), " "),          # any whitespace
]


# -------------- token-counting helpers --------------

def _identify_special_token_ids_from_empty_text_tokenization(clip, stream_key):
    """
    Returns (start_token_id, end_token_id, pad_token_id) by tokenizing the
    empty string and inspecting the first chunk's structure.

    Stock SD1ClipModel emits each chunk as
        [(start, w), <content>, (end, w), (pad, w), ...]
    so the first three IDs from an empty-text chunk give us the markers.
    """
    empty_tokenization_chunks_per_stream = clip.tokenize("")
    if stream_key not in empty_tokenization_chunks_per_stream:
        return (None, None, None)
    first_empty_chunk = empty_tokenization_chunks_per_stream[stream_key][0]
    if len(first_empty_chunk) < 3:
        return (None, None, None)
    start_token_id_value = first_empty_chunk[0][0]
    # In an empty chunk, the very next token is end-of-text (since content is
    # empty). After end, all remaining slots are padding.
    end_token_id_value = first_empty_chunk[1][0]
    pad_token_id_value = first_empty_chunk[-1][0]
    return (start_token_id_value, end_token_id_value, pad_token_id_value)


def _count_content_tokens_in_text_using_clip_tokenizer(text, clip, stream_key):
    """
    Returns the number of CONTENT tokens (excluding start/end/pad) that the
    CLIP tokenizer would produce for `text` on the given stream.

    Used by the balancing algorithm to evaluate how to partition a long
    segment into roughly-equal-sized sub-pieces.
    """
    if not text or not text.strip():
        return 0
    full_tokenization_chunks_per_stream = clip.tokenize(text)
    if stream_key not in full_tokenization_chunks_per_stream:
        return 0
    chunks_list_for_stream = full_tokenization_chunks_per_stream[stream_key]
    if not chunks_list_for_stream:
        return 0

    start_token_id, end_token_id, pad_token_id = _identify_special_token_ids_from_empty_text_tokenization(
        clip, stream_key
    )

    total_full_chunks_before_last_chunk = max(0, len(chunks_list_for_stream) - 1)
    content_tokens_from_full_chunks = (
        total_full_chunks_before_last_chunk * CLIP_CONTENT_TOKENS_PER_CHUNK_EXCLUDING_MARKERS
    )

    last_chunk_in_stream = chunks_list_for_stream[-1]
    last_chunk_content_token_count = 0
    for token_id_in_pair, _weight_in_pair in last_chunk_in_stream:
        if pad_token_id is not None and token_id_in_pair == pad_token_id:
            break
        if start_token_id is not None and token_id_in_pair == start_token_id:
            continue
        if end_token_id is not None and token_id_in_pair == end_token_id:
            continue
        last_chunk_content_token_count += 1

    return content_tokens_from_full_chunks + last_chunk_content_token_count


# -------------- text splitting helpers --------------

def _split_text_at_break_markers_into_hard_segments(text):
    """
    Splits at the BREAK marker (whole word, surrounded by whitespace or
    string-edge). Returns a list of segments — never crosses a BREAK during
    later balancing.
    """
    if not text:
        return [""]
    raw_segments_from_break_split = BREAK_MARKER_REGEX_PATTERN.split(text)
    cleaned_segments_after_stripping_whitespace = [seg.strip() for seg in raw_segments_from_break_split]
    return cleaned_segments_after_stripping_whitespace


def _force_character_split_text_into_fit_guaranteed_pieces(
    text, target_piece_count_as_starting_point, clip, stream_key
):
    """
    Last-resort character-based split. Starts at `target_piece_count_as_starting_point`
    roughly-equal substring slices, then iteratively halves any piece still
    exceeding the per-chunk token budget. Returns pieces that EACH fit within
    one CLIP chunk; may return MORE than the starting count if necessary
    (rare — only for pathological inputs like a single very long URL or
    dense Unicode with no delimiters).
    """
    if not text:
        return [""]
    if target_piece_count_as_starting_point <= 1:
        target_piece_count_as_starting_point = 1
    average_piece_length_in_characters = max(1, len(text) // target_piece_count_as_starting_point)
    pieces_after_initial_character_slicing = []
    for piece_index_in_initial_slicing in range(target_piece_count_as_starting_point):
        slice_start_character_index = piece_index_in_initial_slicing * average_piece_length_in_characters
        if piece_index_in_initial_slicing == target_piece_count_as_starting_point - 1:
            slice_end_character_index = len(text)
        else:
            slice_end_character_index = (piece_index_in_initial_slicing + 1) * average_piece_length_in_characters
        pieces_after_initial_character_slicing.append(text[slice_start_character_index:slice_end_character_index])

    # Iteratively halve any piece that still exceeds the token budget.
    # Bound the number of halving passes to avoid pathological infinite loops.
    maximum_halving_passes_to_attempt = 12
    for _halving_pass_index in range(maximum_halving_passes_to_attempt):
        any_piece_still_oversized_after_this_pass = False
        pieces_after_this_halving_pass = []
        for current_piece_text in pieces_after_initial_character_slicing:
            if not current_piece_text:
                pieces_after_this_halving_pass.append(current_piece_text)
                continue
            current_piece_token_count = _count_content_tokens_in_text_using_clip_tokenizer(
                current_piece_text, clip, stream_key
            )
            if current_piece_token_count <= CLIP_CONTENT_TOKENS_PER_CHUNK_EXCLUDING_MARKERS:
                pieces_after_this_halving_pass.append(current_piece_text)
            else:
                any_piece_still_oversized_after_this_pass = True
                halving_split_character_index = max(1, len(current_piece_text) // 2)
                pieces_after_this_halving_pass.append(current_piece_text[:halving_split_character_index])
                pieces_after_this_halving_pass.append(current_piece_text[halving_split_character_index:])
        pieces_after_initial_character_slicing = pieces_after_this_halving_pass
        if not any_piece_still_oversized_after_this_pass:
            break
    return pieces_after_initial_character_slicing


def _greedy_bin_pack_pieces_into_k_bins_targeting_equal_token_count(
    pieces_list, piece_token_counts_list, target_bin_count
):
    """
    Distributes `pieces_list` into `target_bin_count` bins as
    a list-of-lists, advancing to the next bin once the current bin's
    accumulated token count reaches (total / target_bin_count). Greedy and
    cheap; not optimal but close enough for our needs.
    """
    total_tokens_across_all_pieces = sum(piece_token_counts_list)
    target_tokens_per_bin = (
        total_tokens_across_all_pieces / float(target_bin_count) if target_bin_count > 0 else 0.0
    )
    bins_of_pieces = [[] for _ in range(target_bin_count)]
    bin_running_token_counts = [0] * target_bin_count
    current_bin_index_for_packing = 0
    for piece_text, piece_token_count in zip(pieces_list, piece_token_counts_list):
        bins_of_pieces[current_bin_index_for_packing].append(piece_text)
        bin_running_token_counts[current_bin_index_for_packing] += piece_token_count
        if (
            bin_running_token_counts[current_bin_index_for_packing] >= target_tokens_per_bin
            and current_bin_index_for_packing < target_bin_count - 1
        ):
            current_bin_index_for_packing += 1
    return bins_of_pieces


def _try_to_balance_split_a_single_segment_into_k_pieces_via_delimiter_cascade(
    segment_text, target_piece_count, clip, stream_key
):
    """
    Attempts each delimiter in DELIMITER_CASCADE_FOR_SUBSPLITTING and returns
    the first balanced K-piece split where every resulting piece fits within
    CLIP_CONTENT_TOKENS_PER_CHUNK. Falls back to character slicing if no
    delimiter level produces a successful split.
    """
    if target_piece_count <= 1:
        return [segment_text]

    for delimiter_split_regex, delimiter_rejoin_string in DELIMITER_CASCADE_FOR_SUBSPLITTING_OVERLONG_SEGMENTS:
        candidate_pieces_after_split = [
            piece.strip()
            for piece in delimiter_split_regex.split(segment_text)
            if piece.strip()
        ]
        if len(candidate_pieces_after_split) < target_piece_count:
            continue

        per_piece_token_counts = [
            _count_content_tokens_in_text_using_clip_tokenizer(piece, clip, stream_key)
            for piece in candidate_pieces_after_split
        ]
        bins_of_pieces_after_packing = _greedy_bin_pack_pieces_into_k_bins_targeting_equal_token_count(
            candidate_pieces_after_split, per_piece_token_counts, target_piece_count
        )

        rejoined_bin_texts = [delimiter_rejoin_string.join(bin_pieces).strip() for bin_pieces in bins_of_pieces_after_packing]
        rejoined_bin_texts = [t for t in rejoined_bin_texts if t]

        if len(rejoined_bin_texts) != target_piece_count:
            continue

        all_resulting_bin_texts_fit_within_one_chunk = all(
            _count_content_tokens_in_text_using_clip_tokenizer(text, clip, stream_key)
            <= CLIP_CONTENT_TOKENS_PER_CHUNK_EXCLUDING_MARKERS
            for text in rejoined_bin_texts
        )
        if all_resulting_bin_texts_fit_within_one_chunk:
            return rejoined_bin_texts

    return _force_character_split_text_into_fit_guaranteed_pieces(
        segment_text, target_piece_count, clip, stream_key
    )


def _split_full_text_into_balanced_chunk_sized_pieces_respecting_break_markers(text, clip, stream_key):
    """
    Top-level text-splitter. Returns a list of text pieces where each piece
    is expected to tokenize to ≤ one 77-token CLIP chunk, BREAK markers are
    respected as hard boundaries, and pieces within each BREAK segment are
    roughly balanced in token count.
    """
    if not text or not text.strip():
        return [""]

    hard_segments_separated_by_break_markers = _split_text_at_break_markers_into_hard_segments(text)

    final_text_pieces_ready_to_tokenize = []
    for segment_text in hard_segments_separated_by_break_markers:
        if not segment_text:
            continue

        segment_total_content_token_count = _count_content_tokens_in_text_using_clip_tokenizer(
            segment_text, clip, stream_key
        )
        if segment_total_content_token_count == 0:
            continue

        target_piece_count_for_segment = max(
            1,
            math.ceil(segment_total_content_token_count / float(CLIP_CONTENT_TOKENS_PER_CHUNK_EXCLUDING_MARKERS)),
        )

        if target_piece_count_for_segment == 1:
            final_text_pieces_ready_to_tokenize.append(segment_text)
        else:
            balanced_sub_pieces_for_this_segment = (
                _try_to_balance_split_a_single_segment_into_k_pieces_via_delimiter_cascade(
                    segment_text, target_piece_count_for_segment, clip, stream_key
                )
            )
            final_text_pieces_ready_to_tokenize.extend(balanced_sub_pieces_for_this_segment)

    if not final_text_pieces_ready_to_tokenize:
        return [""]
    return final_text_pieces_ready_to_tokenize


# -------------- concat-pipeline helpers (stock-style multi-chunk encode) --------------

def _tokenize_text_respecting_break_markers_returning_all_chunks_flat(text, clip, stream_key):
    """
    Returns a flat list of CLIP-77-token chunks for `text` on the given
    stream, splitting at BREAK markers so each BREAK forces a new chunk
    boundary. Within each BREAK-segment the CLIP tokenizer's natural greedy
    chunking is used (no balancing — concat mode matches stock behavior).

    If the input text is empty or whitespace-only, returns a single empty
    chunk (the tokenizer's empty-text chunk for this stream).
    """
    if not text or not text.strip():
        empty_chunks_dict = clip.tokenize("")
        return list(empty_chunks_dict[stream_key])

    hard_segments_separated_by_break_markers = _split_text_at_break_markers_into_hard_segments(text)

    flat_chunks_list_concatenated_across_all_segments = []
    for segment_text in hard_segments_separated_by_break_markers:
        if not segment_text:
            continue
        per_segment_chunks_dict = clip.tokenize(segment_text)
        flat_chunks_list_concatenated_across_all_segments.extend(per_segment_chunks_dict[stream_key])

    if not flat_chunks_list_concatenated_across_all_segments:
        empty_chunks_dict = clip.tokenize("")
        return list(empty_chunks_dict[stream_key])
    return flat_chunks_list_concatenated_across_all_segments


def _pad_two_chunk_lists_to_matching_length_with_empty_chunks(
    chunks_list_for_g_stream, chunks_list_for_l_stream, clip
):
    """Stock-style pad-shorter-with-empty so g and l have the same chunk count."""
    empty_tokens_dict = clip.tokenize("")
    while len(chunks_list_for_g_stream) < len(chunks_list_for_l_stream):
        chunks_list_for_g_stream.append(empty_tokens_dict["g"][0])
    while len(chunks_list_for_l_stream) < len(chunks_list_for_g_stream):
        chunks_list_for_l_stream.append(empty_tokens_dict["l"][0])
    return chunks_list_for_g_stream, chunks_list_for_l_stream


def _encode_in_concat_pipeline_returning_single_cond_entry_and_diagnostics(
    g_mode, l_mode, text_g, text_l, clip, sdxl_size_conditioning_add_dict
):
    """
    Stock-style pipeline: build the chunk lists for both streams per their
    mode, pad to matching length, and call encode_from_tokens_scheduled
    ONCE with the full multi-chunk tokens dict. Returns:
        (conditioning, diagnostics_dict)
    where diagnostics_dict has keys:
        - chunk_count_for_g_before_pad
        - chunk_count_for_l_before_pad
        - chunk_count_after_padding
        - break_segment_count_for_g
        - break_segment_count_for_l
        - g_content_token_count
        - l_content_token_count
        - encoded_token_sequence_length
    """
    if g_mode == SPLIT_AND_MERGE_MODE_TRUNCATE:
        g_chunks = clip.tokenize(text_g)["g"][:1]
    else:  # concat
        g_chunks = _tokenize_text_respecting_break_markers_returning_all_chunks_flat(text_g, clip, "g")

    if l_mode == SPLIT_AND_MERGE_MODE_TRUNCATE:
        l_chunks = clip.tokenize(text_l)["l"][:1]
    else:  # concat
        l_chunks = _tokenize_text_respecting_break_markers_returning_all_chunks_flat(text_l, clip, "l")

    chunk_count_for_g_before_pad = len(g_chunks)
    chunk_count_for_l_before_pad = len(l_chunks)

    g_chunks, l_chunks = _pad_two_chunk_lists_to_matching_length_with_empty_chunks(g_chunks, l_chunks, clip)
    chunk_count_after_padding = len(g_chunks)

    tokens_for_single_encode_call = {"g": g_chunks, "l": l_chunks}
    encoded_conditioning_with_all_chunks_concatenated_along_seq_dim = clip.encode_from_tokens_scheduled(
        tokens_for_single_encode_call, add_dict=sdxl_size_conditioning_add_dict
    )

    encoded_token_sequence_length = None
    if (
        encoded_conditioning_with_all_chunks_concatenated_along_seq_dim
        and len(encoded_conditioning_with_all_chunks_concatenated_along_seq_dim) > 0
    ):
        first_entry_tokens_tensor = encoded_conditioning_with_all_chunks_concatenated_along_seq_dim[0][0]
        if hasattr(first_entry_tokens_tensor, "shape") and len(first_entry_tokens_tensor.shape) >= 2:
            encoded_token_sequence_length = int(first_entry_tokens_tensor.shape[1])

    break_segment_count_for_g = len(
        [seg for seg in _split_text_at_break_markers_into_hard_segments(text_g) if seg]
    )
    break_segment_count_for_l = len(
        [seg for seg in _split_text_at_break_markers_into_hard_segments(text_l) if seg]
    )
    g_content_token_count = _count_content_tokens_in_text_using_clip_tokenizer(text_g, clip, "g")
    l_content_token_count = _count_content_tokens_in_text_using_clip_tokenizer(text_l, clip, "l")

    diagnostics_dict_for_caller = {
        "chunk_count_for_g_before_pad": chunk_count_for_g_before_pad,
        "chunk_count_for_l_before_pad": chunk_count_for_l_before_pad,
        "chunk_count_after_padding": chunk_count_after_padding,
        "break_segment_count_for_g": break_segment_count_for_g,
        "break_segment_count_for_l": break_segment_count_for_l,
        "g_content_token_count": g_content_token_count,
        "l_content_token_count": l_content_token_count,
        "encoded_token_sequence_length": encoded_token_sequence_length,
    }
    return (
        encoded_conditioning_with_all_chunks_concatenated_along_seq_dim,
        diagnostics_dict_for_caller,
    )


def _build_concat_or_truncate_pipeline_debug_info_string(
    active_pipeline_label,
    g_mode,
    l_mode,
    diagnostics_dict,
    output_reduction_strategy_label,
    final_conditioning_entry_count,
):
    return (
        "CLIPTextEncodeSDXL (auto-split-and-merge):\n"
        f"  active pipeline: {active_pipeline_label}\n"
        f"  split_and_merge_g: {g_mode}\n"
        f"  split_and_merge_l: {l_mode}\n"
        f"  text_g content tokens: {diagnostics_dict['g_content_token_count']}\n"
        f"  text_l content tokens: {diagnostics_dict['l_content_token_count']}\n"
        f"  BREAK segments (g/l): "
        f"{diagnostics_dict['break_segment_count_for_g']} / "
        f"{diagnostics_dict['break_segment_count_for_l']}\n"
        f"  77-token chunks from tokenizer (g/l before padding): "
        f"{diagnostics_dict['chunk_count_for_g_before_pad']} / "
        f"{diagnostics_dict['chunk_count_for_l_before_pad']}\n"
        f"  77-token chunks after pad-to-match: {diagnostics_dict['chunk_count_after_padding']}\n"
        f"  encoded token sequence length: {diagnostics_dict['encoded_token_sequence_length']}\n"
        f"  output reduction strategy: {output_reduction_strategy_label}\n"
        f"  final CONDITIONING entry count: {final_conditioning_entry_count}"
    )


def _determine_active_pipeline_from_mode_pair(g_mode, l_mode):
    """
    Returns the merge-mode label of the pipeline that should run for the
    given (g_mode, l_mode) pair, per the priority order
    combine > average > concat > truncate.
    """
    for priority_mode_label in SPLIT_AND_MERGE_MODE_PIPELINE_PRIORITY_ORDER_HIGHEST_FIRST:
        if g_mode == priority_mode_label or l_mode == priority_mode_label:
            return priority_mode_label
    return SPLIT_AND_MERGE_MODE_TRUNCATE


# -------------- pair encoding helpers --------------

def _pad_token_chunks_dict_g_and_l_to_matching_length(tokens_dict, clip):
    """
    Mirrors stock CLIPTextEncodeSDXL's pad-shorter-with-empty logic so g and
    l have the same chunk count before encoding.
    """
    empty_tokens_dict = clip.tokenize("")
    while len(tokens_dict["l"]) < len(tokens_dict["g"]):
        tokens_dict["l"] += empty_tokens_dict["l"]
    while len(tokens_dict["l"]) > len(tokens_dict["g"]):
        tokens_dict["g"] += empty_tokens_dict["g"]
    return tokens_dict


def _encode_one_paired_text_piece(text_piece_for_g, text_piece_for_l, clip, sdxl_size_conditioning_add_dict):
    tokens_for_this_pair = clip.tokenize(text_piece_for_g)
    tokens_for_this_pair["l"] = clip.tokenize(text_piece_for_l)["l"]
    tokens_for_this_pair = _pad_token_chunks_dict_g_and_l_to_matching_length(tokens_for_this_pair, clip)
    return clip.encode_from_tokens_scheduled(
        tokens_for_this_pair, add_dict=sdxl_size_conditioning_add_dict
    )


def _average_per_chunk_conditionings_into_a_single_conditioning_entry(per_chunk_conditioning_list):
    if len(per_chunk_conditioning_list) == 0:
        raise ValueError(
            "CLIPTextEncodeSDXLAutoSplitAndMerge: cannot average an empty list of per-chunk conditionings."
        )

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


# -------------- main node class --------------

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
                    {"default": SPLIT_AND_MERGE_MODE_CONCAT},
                ),
                "split_and_merge_l": (
                    SPLIT_AND_MERGE_MODE_CHOICES_IN_DROPDOWN_ORDER,
                    {"default": SPLIT_AND_MERGE_MODE_CONCAT},
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
        sdxl_size_conditioning_add_dict = {
            "width": width,
            "height": height,
            "crop_w": crop_w,
            "crop_h": crop_h,
            "target_width": target_width,
            "target_height": target_height,
        }

        active_pipeline_for_this_invocation = _determine_active_pipeline_from_mode_pair(
            split_and_merge_g, split_and_merge_l
        )

        # ---- CONCAT pipeline: stock-style multi-chunk single-encode-call ----
        if active_pipeline_for_this_invocation == SPLIT_AND_MERGE_MODE_CONCAT:
            output_conditioning_list, concat_pipeline_diagnostics_dict = (
                _encode_in_concat_pipeline_returning_single_cond_entry_and_diagnostics(
                    split_and_merge_g, split_and_merge_l, text_g, text_l, clip, sdxl_size_conditioning_add_dict
                )
            )
            output_reduction_strategy_label = "concat (stock-style seq-dim concat across all chunks)"
            debug_info_string = _build_concat_or_truncate_pipeline_debug_info_string(
                active_pipeline_for_this_invocation,
                split_and_merge_g,
                split_and_merge_l,
                concat_pipeline_diagnostics_dict,
                output_reduction_strategy_label,
                len(output_conditioning_list),
            )
            return (output_conditioning_list, debug_info_string)

        # ---- TRUNCATE pipeline: drop overflow on both sides, single encode ----
        if active_pipeline_for_this_invocation == SPLIT_AND_MERGE_MODE_TRUNCATE:
            output_conditioning_list, truncate_pipeline_diagnostics_dict = (
                _encode_in_concat_pipeline_returning_single_cond_entry_and_diagnostics(
                    SPLIT_AND_MERGE_MODE_TRUNCATE,
                    SPLIT_AND_MERGE_MODE_TRUNCATE,
                    text_g,
                    text_l,
                    clip,
                    sdxl_size_conditioning_add_dict,
                )
            )
            output_reduction_strategy_label = "truncate (first chunk only, both streams)"
            debug_info_string = _build_concat_or_truncate_pipeline_debug_info_string(
                active_pipeline_for_this_invocation,
                split_and_merge_g,
                split_and_merge_l,
                truncate_pipeline_diagnostics_dict,
                output_reduction_strategy_label,
                len(output_conditioning_list),
            )
            return (output_conditioning_list, debug_info_string)

        # ---- COMBINE or AVERAGE pipeline: per-piece balanced split + per-pair encode ----
        text_pieces_for_g_stream = _split_full_text_into_balanced_chunk_sized_pieces_respecting_break_markers(
            text_g, clip, "g"
        )
        text_pieces_for_l_stream = _split_full_text_into_balanced_chunk_sized_pieces_respecting_break_markers(
            text_l, clip, "l"
        )
        initial_piece_count_for_g = len(text_pieces_for_g_stream)
        initial_piece_count_for_l = len(text_pieces_for_l_stream)

        # truncate on a stream reduces that side to first balanced piece.
        # concat on a stream in this pipeline keeps all balanced pieces (we're
        # in combine/average pipeline because the OTHER side picked that).
        if split_and_merge_g == SPLIT_AND_MERGE_MODE_TRUNCATE and len(text_pieces_for_g_stream) > 0:
            text_pieces_for_g_stream = text_pieces_for_g_stream[:1]
        if split_and_merge_l == SPLIT_AND_MERGE_MODE_TRUNCATE and len(text_pieces_for_l_stream) > 0:
            text_pieces_for_l_stream = text_pieces_for_l_stream[:1]

        paired_piece_count_to_encode = max(len(text_pieces_for_g_stream), len(text_pieces_for_l_stream))
        while len(text_pieces_for_g_stream) < paired_piece_count_to_encode:
            text_pieces_for_g_stream.append("")
        while len(text_pieces_for_l_stream) < paired_piece_count_to_encode:
            text_pieces_for_l_stream.append("")

        per_paired_piece_conditioning_list = []
        per_paired_piece_token_count_summary_lines = []
        for paired_piece_index in range(paired_piece_count_to_encode):
            current_text_piece_g = text_pieces_for_g_stream[paired_piece_index]
            current_text_piece_l = text_pieces_for_l_stream[paired_piece_index]
            encoded_pair = _encode_one_paired_text_piece(
                current_text_piece_g, current_text_piece_l, clip, sdxl_size_conditioning_add_dict
            )
            per_paired_piece_conditioning_list.append(encoded_pair)
            per_paired_piece_token_count_summary_lines.append(
                f"  pair[{paired_piece_index}]: "
                f"g_tokens={_count_content_tokens_in_text_using_clip_tokenizer(current_text_piece_g, clip, 'g')}, "
                f"l_tokens={_count_content_tokens_in_text_using_clip_tokenizer(current_text_piece_l, clip, 'l')}"
            )

        if active_pipeline_for_this_invocation == SPLIT_AND_MERGE_MODE_COMBINE:
            output_conditioning_list = []
            for per_pair_conditioning in per_paired_piece_conditioning_list:
                output_conditioning_list.extend(per_pair_conditioning)
            output_reduction_strategy_label = "combine (parallel branches)"
        else:  # AVERAGE
            averaged_single_entry = _average_per_chunk_conditionings_into_a_single_conditioning_entry(
                per_paired_piece_conditioning_list
            )
            output_conditioning_list = [averaged_single_entry]
            output_reduction_strategy_label = "average (blended embeddings)"

        debug_info_string = (
            "CLIPTextEncodeSDXL (auto-split-and-merge):\n"
            f"  active pipeline: {active_pipeline_for_this_invocation}\n"
            f"  text_g balanced piece count: {initial_piece_count_for_g}\n"
            f"  text_l balanced piece count: {initial_piece_count_for_l}\n"
            f"  split_and_merge_g: {split_and_merge_g}\n"
            f"  split_and_merge_l: {split_and_merge_l}\n"
            f"  paired pieces actually encoded: {paired_piece_count_to_encode}\n"
            + "\n".join(per_paired_piece_token_count_summary_lines)
            + f"\n  output reduction strategy: {output_reduction_strategy_label}\n"
            f"  final CONDITIONING entry count: {len(output_conditioning_list)}"
        )

        return (output_conditioning_list, debug_info_string)
