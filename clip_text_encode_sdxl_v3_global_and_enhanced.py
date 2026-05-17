"""
CLIP Text Encode SDXL v3 (Global + Enhanced)

A sibling of v1's CLIPTextEncodeWithCutoffRegionSeparation that more
faithfully mirrors BlenderNeko's BNK_CutoffRegionsToConditioning two-text-
per-region model (target_text vs region_text), but with the per-section
structure flipped to be ergonomic:

Per section:
  - global_text (left) — a chunk of the user's natural prompt. Sequential
    global_text fields comma-joined form the "user-visible plain prompt".
    Also serves as cutoff's TARGET_TEXT: space-split into words, each word
    becomes a target that gets masked in OTHER sections' encoding passes.
  - enhanced_text (right, optional) — an EXPANDED version of global_text
    that's what the model actually encodes at this section's position.
    Plays cutoff's REGION_TEXT role. Must contain all of global_text's
    words somewhere within (validated; warnings surfaced via the realtime
    embedding-error widget).

If enhanced_text is empty (or equals global_text), the section is a
PASSTHROUGH chunk: it contributes its tokens to the base prompt but
creates no region/target. No per-region encoding pass, no overlay
contribution for that section.

The actual prompt the model encodes (the "base prompt") is the comma-join
of each section's enhanced_text (or global_text if enhanced is empty/equal).

Encoding runs per CLIP stream (L and G) independently. Each section has
per-stream strength FLOAT widgets; strength = 0 excludes that section
from that stream entirely (no `(tag:0)` paradox).

Hard-coded cutoff knobs: strict_mask = 1.0, start_from_masked = 1.0
(v1's exposure of these as widgets was confusing and the user found
1.0/1.0 worked well in practice — kept as defaults, not exposed).

NO external plugin dependency. Self-contained. v1 is untouched.
"""

import logging
import os
import re

import numpy as np
import torch

import folder_paths

from .cutoff_per_stream_isolation import (
    _get_per_stream_clip_text_encoder_model_and_tokenizer,
    _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad,
    _find_all_sublist_match_start_positions_within_superlist,
    _expand_content_position_mask_into_full_chunk_positions_with_start_end_padding,
    _build_masked_tokens_per_chunk_by_replacing_target_positions_with_mask_token_id,
    _resolve_mask_token_id_from_user_string_with_default,
    _reshape_any_mismatched_embedding_tensors_in_chunks_list_in_place_to_match_stream_expected_dim,
    CHUNK_CONTENT_TOKEN_COUNT_EXCLUDING_MARKERS,
)


MAX_SECTION_COUNT_SUPPORTED = 32
DEFAULT_SECTION_COUNT_VALUE = 2

# Hard-coded cutoff-style knobs (not exposed as widgets — v1's exposure
# of these confused users and the user reported 1.0/1.0 works best).
HARDCODED_STRICT_MASK_VALUE = 1.0
HARDCODED_START_FROM_MASKED_VALUE = 1.0
HARDCODED_MASK_TOKEN_STRING_USE_END_OF_TEXT_DEFAULT = ""

LATENT_TO_IMAGE_SPATIAL_SCALE_FACTOR = 8
DEFAULT_LATENT_IMAGE_WIDTH_WHEN_NO_LATENT_INPUT = 1024
DEFAULT_LATENT_IMAGE_HEIGHT_WHEN_NO_LATENT_INPUT = 1024

ZOOM_MINIMUM_VALUE = 1.0
ZOOM_MAXIMUM_VALUE = 100.0
ZOOM_DEFAULT_VALUE = 1.0
OFFSET_MINIMUM_VALUE = -1.0
OFFSET_MAXIMUM_VALUE = 1.0
OFFSET_DEFAULT_VALUE = 0.0

UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE = 1.0
UPSCALED_CONDITIONING_MULTIPLIER_DEFAULT_VALUE = 1.0

WHITESPACE_RUN_REGEX_FOR_NORMALIZING_SECTION_TEXT = re.compile(r"\s+")


# -------------- section collection --------------

def _emit_stock_style_shape_mismatch_warning_for_each_embedding_in_text_that_will_not_match_its_clip_stream_dim(
    clip_object_to_inspect, prompt_text_to_scan_for_embedding_references
):
    """
    Mirror the WARNING emitted by ComfyUI's stock CLIPTextEncode when a
    textual-inversion embedding's saved hidden-dim does not match a CLIP
    stream's expected hidden-dim (e.g. an SD1.5 768-dim embedding referenced
    in an SDXL prompt where CLIP-G expects 1280).

    Stock's warning comes from `SDClipModel.process_tokens` (sd1_clip.py)
    only when the per-stream encode runs. Our node's per-stream chunking /
    cutoff path appears to (in some workflows) consume or transform the
    tokenized result before that warning would fire, so we proactively
    detect the same condition by tokenizing the raw prompt text the same
    way stock would (`clip.tokenize`) and inspecting the per-stream chunks
    for tensor-typed token entries whose `shape[-1]` does not match that
    stream's `embedding_size`.

    Emits one `logging.warning` per mismatched embedding occurrence, using
    the EXACT format string stock uses so existing user expectations and
    log greps continue to work:
        "WARNING: shape mismatch when trying to apply embedding,
         embedding will be ignored {emb_dim} != {expected_dim}"
    """
    if clip_object_to_inspect is None or not prompt_text_to_scan_for_embedding_references:
        return
    try:
        per_stream_tokenized_chunks_keyed_by_stream_name = clip_object_to_inspect.tokenize(
            prompt_text_to_scan_for_embedding_references
        )
    except Exception:
        return
    if not isinstance(per_stream_tokenized_chunks_keyed_by_stream_name, dict):
        return
    top_level_tokenizer_object_or_none = getattr(clip_object_to_inspect, "tokenizer", None)
    for stream_name_key, chunks_list_for_this_stream in per_stream_tokenized_chunks_keyed_by_stream_name.items():
        if not isinstance(chunks_list_for_this_stream, list):
            continue
        per_stream_tokenizer_object_or_none = (
            getattr(top_level_tokenizer_object_or_none, stream_name_key, None)
            if top_level_tokenizer_object_or_none is not None
            else None
        )
        expected_hidden_dim_for_this_stream = getattr(
            per_stream_tokenizer_object_or_none, "embedding_size", None
        )
        if expected_hidden_dim_for_this_stream is None:
            continue
        for one_chunk_of_token_weight_pairs in chunks_list_for_this_stream:
            if not isinstance(one_chunk_of_token_weight_pairs, list):
                continue
            for pair_in_chunk in one_chunk_of_token_weight_pairs:
                if not isinstance(pair_in_chunk, (list, tuple)) or len(pair_in_chunk) < 1:
                    continue
                token_value_in_pair = pair_in_chunk[0]
                if not isinstance(token_value_in_pair, torch.Tensor):
                    continue
                actual_embedding_hidden_dim = token_value_in_pair.shape[-1]
                if actual_embedding_hidden_dim != expected_hidden_dim_for_this_stream:
                    logging.warning(
                        "WARNING: shape mismatch when trying to apply embedding, "
                        "embedding will be ignored {} != {}".format(
                            actual_embedding_hidden_dim,
                            expected_hidden_dim_for_this_stream,
                        )
                    )


def _normalize_whitespace_in_user_supplied_text(raw_text):
    return WHITESPACE_RUN_REGEX_FOR_NORMALIZING_SECTION_TEXT.sub(" ", (raw_text or "")).strip()


def _collect_active_v3_section_descriptors_from_kwargs_in_declaration_order(
    kwargs_with_per_section_widget_values, active_section_count_setting_value
):
    """
    Reads each section's six widgets:
      - section_N_global_text          (STRING multiline)
      - section_N_enhanced_text        (STRING multiline)
      - section_N_global_text_weight   (FLOAT)
      - section_N_enhanced_text_weight (FLOAT)
      - section_N_clip_l_strength      (FLOAT, 0 = exclude from L)
      - section_N_clip_g_strength      (FLOAT, 0 = exclude from G)

    Skips a section entirely if BOTH global_text and enhanced_text are
    empty after whitespace normalization (nothing to contribute).

    Sets `is_true_region` = True iff enhanced_text is non-empty AND differs
    (after case-insensitive whitespace-normalized comparison) from
    global_text. True-region sections get cutoff per-region encoding;
    others are passthrough chunks.
    """
    collected_section_descriptor_list_in_declaration_order = []
    for one_based_section_index in range(1, int(active_section_count_setting_value) + 1):
        normalized_global_text_for_this_section = _normalize_whitespace_in_user_supplied_text(
            kwargs_with_per_section_widget_values.get(f"section_{one_based_section_index}_global_text", "")
        )
        normalized_enhanced_text_for_this_section = _normalize_whitespace_in_user_supplied_text(
            kwargs_with_per_section_widget_values.get(f"section_{one_based_section_index}_enhanced_text", "")
        )
        if not normalized_global_text_for_this_section and not normalized_enhanced_text_for_this_section:
            continue
        enhanced_is_meaningfully_different_from_global = (
            bool(normalized_enhanced_text_for_this_section)
            and normalized_enhanced_text_for_this_section.lower() != normalized_global_text_for_this_section.lower()
        )
        collected_section_descriptor_list_in_declaration_order.append({
            "section_id_one_based": one_based_section_index,
            "global_text": normalized_global_text_for_this_section,
            "enhanced_text": normalized_enhanced_text_for_this_section,
            "global_text_weight": float(
                kwargs_with_per_section_widget_values.get(
                    f"section_{one_based_section_index}_global_text_weight", 1.0
                )
            ),
            "enhanced_text_weight": float(
                kwargs_with_per_section_widget_values.get(
                    f"section_{one_based_section_index}_enhanced_text_weight", 1.0
                )
            ),
            "clip_l_strength": float(
                kwargs_with_per_section_widget_values.get(
                    f"section_{one_based_section_index}_clip_l_strength", 1.0
                )
            ),
            "clip_g_strength": float(
                kwargs_with_per_section_widget_values.get(
                    f"section_{one_based_section_index}_clip_g_strength", 1.0
                )
            ),
            "is_true_region": enhanced_is_meaningfully_different_from_global,
        })
    return collected_section_descriptor_list_in_declaration_order


def _select_per_stream_strength_value_for_section(section_descriptor, stream_key_l_or_g):
    if stream_key_l_or_g == "l":
        return section_descriptor["clip_l_strength"]
    return section_descriptor["clip_g_strength"]


def _build_per_section_base_prompt_fragment_for_one_stream(section_descriptor, stream_key_l_or_g):
    """
    Returns the `(text:weight)` fragment this section contributes to the
    base prompt for the given stream, or None if the section is excluded
    from this stream (stream strength == 0).

    If the section is a true region: use enhanced_text wrapped with
    enhanced_text_weight × stream_strength.
    If the section is a passthrough: use global_text wrapped with
    global_text_weight × stream_strength.
    If the chosen text is empty for some reason, returns None.
    """
    per_stream_strength_value = _select_per_stream_strength_value_for_section(
        section_descriptor, stream_key_l_or_g
    )
    if per_stream_strength_value == 0:
        return None
    if section_descriptor["is_true_region"]:
        text_to_wrap = section_descriptor["enhanced_text"]
        weight_to_apply = section_descriptor["enhanced_text_weight"] * per_stream_strength_value
    else:
        text_to_wrap = section_descriptor["global_text"] or section_descriptor["enhanced_text"]
        weight_to_apply = section_descriptor["global_text_weight"] * per_stream_strength_value
    if not text_to_wrap:
        return None
    return f"({text_to_wrap}:{round(weight_to_apply, 4)})"


def _pad_per_stream_tokenization_to_match_chunk_count(
    clip_object, embedding_tensor_for_short_stream, target_chunk_count, stream_key_to_pad_with_empty
):
    """
    If a stream's tokenization produced fewer chunks than the other stream,
    SDXL's combine requires matching chunk counts. We pad with empty-prompt
    encodings of the SHORT stream. Returns the padded embedding tensor.

    For now we return the tensor unchanged and let the SDXL combine
    truncate via min() — chunk-count mismatch is a corner case and most
    prompts fit in one chunk for both streams.
    """
    return embedding_tensor_for_short_stream


EMBEDDING_REFERENCE_IN_PROMPT_TEXT_REGEX_PATTERN = re.compile(r"embedding:([\w./\\-]+)")

KNOWN_A1111_EMBEDDING_NAMES_LIST_FILE_RELATIVE_PATH_FROM_THIS_MODULE = (
    "known_a1111_embedding_names_to_filter_when_not_installed_locally.txt"
)

_cached_lowercase_set_of_known_a1111_embedding_names_to_filter = None


def _load_known_a1111_embedding_names_to_filter_into_lowercase_set_from_text_file(
    text_file_absolute_path,
):
    loaded_lowercase_names_set = set()
    try:
        with open(text_file_absolute_path, "r", encoding="utf-8") as opened_text_file_handle:
            for one_line_in_file in opened_text_file_handle:
                stripped_line_text = one_line_in_file.strip()
                if not stripped_line_text or stripped_line_text.startswith("#"):
                    continue
                loaded_lowercase_names_set.add(stripped_line_text.lower())
    except OSError:
        pass
    return loaded_lowercase_names_set


def _get_cached_or_lazily_load_known_a1111_embedding_names_lowercase_set():
    global _cached_lowercase_set_of_known_a1111_embedding_names_to_filter
    if _cached_lowercase_set_of_known_a1111_embedding_names_to_filter is None:
        this_module_directory_absolute_path = os.path.dirname(os.path.abspath(__file__))
        names_list_file_absolute_path = os.path.join(
            this_module_directory_absolute_path,
            KNOWN_A1111_EMBEDDING_NAMES_LIST_FILE_RELATIVE_PATH_FROM_THIS_MODULE,
        )
        _cached_lowercase_set_of_known_a1111_embedding_names_to_filter = (
            _load_known_a1111_embedding_names_to_filter_into_lowercase_set_from_text_file(
                names_list_file_absolute_path
            )
        )
        logging.info(
            f"unified-conditioning-merge: loaded "
            f"{len(_cached_lowercase_set_of_known_a1111_embedding_names_to_filter)} known A1111 "
            f"embedding name(s) from {names_list_file_absolute_path}"
        )
    return _cached_lowercase_set_of_known_a1111_embedding_names_to_filter


def _strip_orphan_a1111_bare_tags_matching_known_names_list_but_not_installed_locally(
    prompt_text_string,
    available_embedding_lowercase_stem_to_filenames_map,
):
    """
    Walks comma-separated tags. For each bare tag (not already
    `embedding:...`) that matches a name in the curated
    known-A1111-embedding-names list (loaded from
    known_a1111_embedding_names_to_filter_when_not_installed_locally.txt
    at the plugin root) AND does NOT correspond to a file installed
    in the user's embeddings folder, drops the tag from the prompt
    entirely. Logs each removal.
    """
    if not prompt_text_string:
        return prompt_text_string

    known_a1111_embedding_names_lowercase_set = (
        _get_cached_or_lazily_load_known_a1111_embedding_names_lowercase_set()
    )
    if not known_a1111_embedding_names_lowercase_set:
        return prompt_text_string

    surviving_comma_separated_parts_list = []
    for raw_comma_separated_part_text in prompt_text_string.split(","):
        stripped_part_text = raw_comma_separated_part_text.strip()
        if not stripped_part_text:
            continue
        # Skip explicit embedding: refs — those are handled by other passes.
        if stripped_part_text.lower().startswith("embedding:"):
            surviving_comma_separated_parts_list.append(raw_comma_separated_part_text)
            continue
        # Unwrap optional `(tag:weight)` paren wrapping for matching purposes.
        bare_tag_text_for_matching = stripped_part_text
        if bare_tag_text_for_matching.startswith("(") and bare_tag_text_for_matching.endswith(")"):
            bare_tag_text_for_matching = bare_tag_text_for_matching[1:-1].strip()
            if ":" in bare_tag_text_for_matching:
                bare_tag_text_for_matching = bare_tag_text_for_matching.rsplit(":", 1)[0].strip()
        if bare_tag_text_for_matching.lower() not in known_a1111_embedding_names_lowercase_set:
            surviving_comma_separated_parts_list.append(raw_comma_separated_part_text)
            continue
        if bare_tag_text_for_matching.lower() in available_embedding_lowercase_stem_to_filenames_map:
            # Tag matches known name AND is installed locally — let it through;
            # the A1111-rewrite pass will turn it into a proper embedding ref.
            surviving_comma_separated_parts_list.append(raw_comma_separated_part_text)
            continue
        logging.info(
            f"Removed orphan A1111-style embedding tag '{bare_tag_text_for_matching}' from prompt "
            f"(matches known-embedding-names list but not installed locally; "
            f"filter_known_a1111_embedding_tags_not_installed_locally is enabled)."
        )
    return ", ".join(part.strip() for part in surviving_comma_separated_parts_list if part.strip())

A1111_STYLE_OPTIONAL_WEIGHT_PARENTHESIZED_TAG_REGEX_PATTERN = re.compile(
    r"^\(\s*([^():,]+?)\s*(?::\s*([0-9.]+)\s*)?\)$"
)


def _build_lookup_of_available_embedding_stems_to_their_filename_list():
    """
    Returns {lowercased_stem: [filename, filename, ...]} mapping the basename-
    without-extension of every file in ComfyUI's `embeddings` directory to the
    full filename(s) that share that stem. Case-insensitive on the key side so
    A1111-style tags (which are usually typed case-loose) can match.
    """
    try:
        available_embedding_filenames_list = folder_paths.get_filename_list("embeddings")
    except Exception:
        return {}
    lowercased_stem_to_filename_list_map = {}
    for embedding_filename in available_embedding_filenames_list:
        stem_without_extension = os.path.splitext(embedding_filename)[0]
        lowercased_stem_to_filename_list_map.setdefault(stem_without_extension.lower(), []).append(
            embedding_filename
        )
    return lowercased_stem_to_filename_list_map


def _rewrite_a1111_style_bare_embedding_tags_to_comfyui_embedding_prefix_form(prompt_text_string):
    """
    Scans the prompt's comma-separated tags. For each tag that EXACTLY matches
    (case-insensitive) the stem of an available embedding file, rewrites it to
    `embedding:STEM` so ComfyUI's tokenizer will load it. Supports the
    `(tag:weight)` parenthesized-weight form too.

    Logs:
      - INFO line per identified tag: which embedding file was matched.
      - WARNING line if the tag matches multiple available files (different
        extensions of the same stem). The first file is used.

    Returns the rewritten prompt text. If no rewrites apply, returns the
    original text unchanged.
    """
    if not prompt_text_string:
        return prompt_text_string

    lowercased_stem_to_filename_list_map = _build_lookup_of_available_embedding_stems_to_their_filename_list()
    if not lowercased_stem_to_filename_list_map:
        return prompt_text_string

    rewritten_comma_separated_parts = []
    for raw_comma_separated_part_with_possible_surrounding_whitespace in prompt_text_string.split(","):
        stripped_part = raw_comma_separated_part_with_possible_surrounding_whitespace.strip()
        if not stripped_part:
            rewritten_comma_separated_parts.append(raw_comma_separated_part_with_possible_surrounding_whitespace)
            continue

        # Strip a single layer of optional `(tag:weight)` parens so the bare
        # tag inside can be matched. If matched, we reassemble with the same
        # parens + weight after substitution.
        bare_token_text_to_consider_for_matching = stripped_part
        opening_paren_for_reassembly = ""
        closing_paren_for_reassembly = ""
        weight_suffix_for_reassembly = ""
        parenthesized_form_match = A1111_STYLE_OPTIONAL_WEIGHT_PARENTHESIZED_TAG_REGEX_PATTERN.match(stripped_part)
        if parenthesized_form_match is not None:
            bare_token_text_to_consider_for_matching = parenthesized_form_match.group(1).strip()
            opening_paren_for_reassembly = "("
            closing_paren_for_reassembly = ")"
            parsed_weight_string_or_none = parenthesized_form_match.group(2)
            if parsed_weight_string_or_none is not None:
                weight_suffix_for_reassembly = f":{parsed_weight_string_or_none}"

        # If it already uses the embedding: prefix, leave it alone.
        if bare_token_text_to_consider_for_matching.lower().startswith("embedding:"):
            rewritten_comma_separated_parts.append(raw_comma_separated_part_with_possible_surrounding_whitespace)
            continue

        matching_filename_list_for_this_tag = lowercased_stem_to_filename_list_map.get(
            bare_token_text_to_consider_for_matching.lower()
        )
        if not matching_filename_list_for_this_tag:
            rewritten_comma_separated_parts.append(raw_comma_separated_part_with_possible_surrounding_whitespace)
            continue

        chosen_embedding_filename = matching_filename_list_for_this_tag[0]
        chosen_embedding_filename_stem = os.path.splitext(chosen_embedding_filename)[0]

        if len(matching_filename_list_for_this_tag) > 1:
            logging.warning(
                f"A1111-style embedding tag '{bare_token_text_to_consider_for_matching}' matches "
                f"multiple files in the embeddings directory: {matching_filename_list_for_this_tag}. "
                f"Using first found: '{chosen_embedding_filename}'."
            )
        logging.info(
            f"A1111-style embedding tag detected: '{bare_token_text_to_consider_for_matching}' "
            f"(no 'embedding:' prefix); rewriting to 'embedding:{chosen_embedding_filename_stem}' "
            f"(matched file: '{chosen_embedding_filename}')."
        )

        rewritten_with_comfyui_prefix_form = (
            f"{opening_paren_for_reassembly}"
            f"embedding:{chosen_embedding_filename_stem}"
            f"{weight_suffix_for_reassembly}"
            f"{closing_paren_for_reassembly}"
        )
        rewritten_comma_separated_parts.append(rewritten_with_comfyui_prefix_form)

    return ", ".join(part.strip() for part in rewritten_comma_separated_parts if part.strip())

# Per-stream expected embedding-vector dim for SDXL CLIP. Non-SDXL conditioning
# (e.g., SD1.5 with only an "l" stream of 768) is handled by the per-stream
# check: an extra "g" stream simply won't be present in the tokens dict.
EXPECTED_EMBEDDING_DIM_PER_STREAM_KEY_FOR_SDXL = {
    "l": 768,
    "g": 1280,
}


def _warn_about_each_mismatched_shape_embedding_reference_in_prompt_text(prompt_text_string, clip_object):
    """
    Scans the prompt text for `embedding:NAME` references. For each one,
    tokenizes it in isolation and inspects the resulting tokens for each
    CLIP stream. If the embedding's loaded tensor has a last-dim that does
    not match the stream's expected embedding dim, prints a warning naming
    the embedding and the actual vs expected dims so the user knows which
    embedding was the mismatched one.

    This is informational only — we do NOT modify or skip the encode. The
    stock encoder downstream will ignore mismatched embeddings (replacing
    them with empty/PAD tokens) and emit its own short warning; this helper
    just gives the user the FILE NAME (which stock's warning omits).
    """
    if not prompt_text_string:
        return
    embedding_names_already_warned_for_per_stream = set()
    for embedding_reference_match in EMBEDDING_REFERENCE_IN_PROMPT_TEXT_REGEX_PATTERN.finditer(prompt_text_string):
        embedding_name_from_prompt = embedding_reference_match.group(1)
        try:
            isolated_tokenization_for_this_one_embedding = clip_object.tokenize(
                f"embedding:{embedding_name_from_prompt}"
            )
        except Exception:
            continue
        for stream_key, expected_dim in EXPECTED_EMBEDDING_DIM_PER_STREAM_KEY_FOR_SDXL.items():
            chunks_list_for_this_stream = isolated_tokenization_for_this_one_embedding.get(stream_key)
            if not chunks_list_for_this_stream:
                continue
            mismatch_already_detected_for_this_stream_for_this_embedding = False
            for one_chunk_of_token_weight_pairs in chunks_list_for_this_stream:
                if mismatch_already_detected_for_this_stream_for_this_embedding:
                    break
                for token_id_or_embedding_tensor, _weight in one_chunk_of_token_weight_pairs:
                    if isinstance(token_id_or_embedding_tensor, torch.Tensor):
                        actual_last_dim_of_embedding_tensor = token_id_or_embedding_tensor.shape[-1]
                        if actual_last_dim_of_embedding_tensor != expected_dim:
                            dedup_key_for_this_specific_warning = (
                                embedding_name_from_prompt,
                                stream_key,
                                actual_last_dim_of_embedding_tensor,
                                expected_dim,
                            )
                            if dedup_key_for_this_specific_warning not in embedding_names_already_warned_for_per_stream:
                                logging.warning(
                                    f"Warning: '{embedding_name_from_prompt}' embedding mismatch detected "
                                    f"on CLIP-{stream_key.upper()} stream "
                                    f"(got dim {actual_last_dim_of_embedding_tensor}, "
                                    f"model expects dim {expected_dim}). "
                                    f"Embedding may be designed for another model. "
                                    f"This node will pad/truncate it to fit so the embedding still contributes "
                                    f"some signal (rather than being silently dropped), but its semantic content "
                                    f"will be distorted. Results may be unexpected."
                                )
                                embedding_names_already_warned_for_per_stream.add(dedup_key_for_this_specific_warning)
                            mismatch_already_detected_for_this_stream_for_this_embedding = True
                            break


_EMBEDDING_REFERENCE_AS_A_COMMA_PART_REGEX_PATTERN = re.compile(
    r"^\(?\s*embedding:([\w./\\-]+)\s*(?::\s*[0-9.]+\s*)?\)?$"
)


def _detect_whether_any_stream_has_a_shape_mismatched_tensor_for_this_embedding(
    embedding_name_string, clip_object
):
    """
    Tokenizes `embedding:NAME` in isolation and checks per stream whether
    the loaded embedding tensor's last dim matches that stream's expected
    embedding_size. Returns True if ANY stream mismatches.

    Bugfix history: an earlier implementation tried to look up the per-
    stream tokenizer via getattr(clip.tokenizer, stream_key) where
    stream_key was 'l' or 'g'. SDXLTokenizer exposes its sub-tokenizers
    as `clip_l` and `clip_g`, so that getattr returned None and the
    expected_dim fell through to None, the check was skipped, and the
    function always returned False — leaving the unsupported-strip path
    a no-op. Now uses the SDXL-hardcoded dim map for consistency with
    the named-warning helper that already does the same.
    """
    try:
        isolated_tokenization_per_stream = clip_object.tokenize(f"embedding:{embedding_name_string}")
    except Exception:
        return False
    if not isinstance(isolated_tokenization_per_stream, dict):
        return False
    for stream_name_key, expected_hidden_dim_for_this_stream in EXPECTED_EMBEDDING_DIM_PER_STREAM_KEY_FOR_SDXL.items():
        chunks_list_for_this_stream = isolated_tokenization_per_stream.get(stream_name_key)
        if not isinstance(chunks_list_for_this_stream, list):
            continue
        for one_chunk_of_token_weight_pairs in chunks_list_for_this_stream:
            if not isinstance(one_chunk_of_token_weight_pairs, list):
                continue
            for token_id_or_embedding_tensor, _weight in one_chunk_of_token_weight_pairs:
                if isinstance(token_id_or_embedding_tensor, torch.Tensor):
                    if token_id_or_embedding_tensor.shape[-1] != expected_hidden_dim_for_this_stream:
                        return True
    return False


def _strip_unsupported_embedding_references_from_prompt_text(prompt_text_string, clip_object):
    """
    Walks the comma-separated parts of `prompt_text_string`. For any part that
    is exactly an `embedding:NAME` reference (with or without parens and
    `:weight`), checks whether the embedding's tensor shape matches the
    current model's CLIP stream(s). If ANY stream mismatches, drops that
    part entirely from the prompt and logs an INFO line.

    Returns the modified prompt text. Embeddings that fit all streams pass
    through untouched, as do non-embedding tags.
    """
    if not prompt_text_string:
        return prompt_text_string

    surviving_comma_separated_parts = []
    for raw_comma_separated_part in prompt_text_string.split(","):
        stripped_part_text_for_pattern_match = raw_comma_separated_part.strip()
        if not stripped_part_text_for_pattern_match:
            continue
        embedding_reference_match = _EMBEDDING_REFERENCE_AS_A_COMMA_PART_REGEX_PATTERN.match(
            stripped_part_text_for_pattern_match
        )
        if embedding_reference_match is None:
            surviving_comma_separated_parts.append(raw_comma_separated_part)
            continue
        embedding_name_from_this_reference = embedding_reference_match.group(1)
        if _detect_whether_any_stream_has_a_shape_mismatched_tensor_for_this_embedding(
            embedding_name_from_this_reference, clip_object
        ):
            logging.info(
                f"Removed unsupported embedding 'embedding:{embedding_name_from_this_reference}' "
                f"from prompt because its tensor shape does not match this model's CLIP stream(s) "
                f"(remove_text_for_unsupported_embeddings is enabled)."
            )
            continue
        surviving_comma_separated_parts.append(raw_comma_separated_part)

    return ", ".join(part.strip() for part in surviving_comma_separated_parts if part.strip())


def _apply_v3_per_text_transforms_to_one_text_string(
    raw_text_string,
    clip_object,
    support_a1111_style_embedding_text_setting,
    remove_text_for_unsupported_embeddings_setting,
    filter_known_a1111_embedding_tags_not_installed_locally_setting,
):
    """
    Apply the same suite of text-transform passes v1 runs on each
    section's text (orphan-A1111-tag filter, A1111 bare-tag rewrite,
    shape-mismatch warning, unsupported-embedding strip), to a single
    text string in isolation. Same order v1 uses.

    The orphan-tag filter reads its names list from the curated
    text file `known_a1111_embedding_names_to_filter_when_not_installed_locally.txt`
    at the plugin root — no per-node override.
    """
    working_text = raw_text_string or ""
    if not working_text.strip():
        return ""
    if filter_known_a1111_embedding_tags_not_installed_locally_setting:
        available_embedding_lowercase_stem_to_filenames_map = (
            _build_lookup_of_available_embedding_stems_to_their_filename_list()
        )
        working_text = (
            _strip_orphan_a1111_bare_tags_matching_known_names_list_but_not_installed_locally(
                working_text,
                available_embedding_lowercase_stem_to_filenames_map,
            )
        )
    if support_a1111_style_embedding_text_setting:
        working_text = _rewrite_a1111_style_bare_embedding_tags_to_comfyui_embedding_prefix_form(
            working_text
        )
    _warn_about_each_mismatched_shape_embedding_reference_in_prompt_text(working_text, clip_object)
    if remove_text_for_unsupported_embeddings_setting:
        working_text = _strip_unsupported_embedding_references_from_prompt_text(
            working_text, clip_object
        )
    _emit_stock_style_shape_mismatch_warning_for_each_embedding_in_text_that_will_not_match_its_clip_stream_dim(
        clip_object, working_text
    )
    return working_text


def _build_per_stream_base_prompt_text_and_per_section_base_fragment_list(
    active_section_descriptors_list, stream_key_l_or_g
):
    """
    Walks the active sections in declaration order, builds each section's
    base-prompt fragment for this stream (or None if excluded by zero
    stream-strength), and returns:
      - the comma-joined base prompt text for this stream
      - the parallel list `[fragment_or_none_for_section_0, ...]` so
        callers can correlate per-section identity with positions in
        the base prompt.
    """
    per_section_base_fragment_or_none_list = []
    base_prompt_text_parts_for_this_stream = []
    for section_descriptor in active_section_descriptors_list:
        per_section_fragment_or_none = _build_per_section_base_prompt_fragment_for_one_stream(
            section_descriptor, stream_key_l_or_g
        )
        per_section_base_fragment_or_none_list.append(per_section_fragment_or_none)
        if per_section_fragment_or_none is not None:
            base_prompt_text_parts_for_this_stream.append(per_section_fragment_or_none)
    return (
        ", ".join(base_prompt_text_parts_for_this_stream),
        per_section_base_fragment_or_none_list,
    )


def _compute_target_words_as_case_insensitive_set_difference_of_enhanced_minus_global(
    section_descriptor,
):
    """
    Target derivation (v3 final design):

      target words = (whitespace-split words in enhanced_text)
                   MINUS (whitespace-split words in global_text)

    The premise: enhanced_text is an EXPANDED version of global_text.
    Words present in enhanced but NOT in global are the "added
    descriptive enhancements" — they're the strong-bias content that
    should be MASKED in other sections' encoding passes so they don't
    contaminate other regions.

    Match is case-insensitive. Each target word is returned in its
    original case from enhanced_text (for downstream tokenization).
    Order is enhanced_text declaration order (so subsequent token
    lookups are deterministic). Duplicates collapsed.

    Returns: list of unique target word strings.
    """
    global_text_lowercase_word_set = set(
        word_lower for word_lower in (section_descriptor.get("global_text") or "").lower().split()
    )
    seen_lowercase_target_words_set = set()
    target_words_in_original_case_in_declaration_order = []
    for one_enhanced_word_original_case in (section_descriptor.get("enhanced_text") or "").split():
        enhanced_word_stripped = one_enhanced_word_original_case.strip()
        if not enhanced_word_stripped:
            continue
        enhanced_word_lowercase = enhanced_word_stripped.lower()
        if enhanced_word_lowercase in global_text_lowercase_word_set:
            continue
        if enhanced_word_lowercase in seen_lowercase_target_words_set:
            continue
        seen_lowercase_target_words_set.add(enhanced_word_lowercase)
        target_words_in_original_case_in_declaration_order.append(enhanced_word_stripped)
    return target_words_in_original_case_in_declaration_order


def _build_per_section_target_mask_over_content_positions_one_stream(
    section_descriptor,
    base_content_token_ids_flat_list,
    expected_content_position_count_across_all_chunks,
    per_stream_tokenizer,
    end_token_id_for_this_stream,
    stream_key_l_or_g,
    clip_object,
):
    """
    Cutoff-style target mask construction for a single section on a
    single stream.

    Target derivation: enhanced_text words MINUS global_text words
    (case-insensitive set difference). See
    `_compute_target_words_as_case_insensitive_set_difference_of_enhanced_minus_global`.

    For each derived target word:
      - tokenize it standalone via this stream's tokenizer
      - find ALL sublist matches within the base prompt's content tokens
      - mark those positions in the target mask

    Returns:
      (target_mask_one_dim_int_array_over_content_positions,
       list_of_target_words_NOT_found_in_base_for_warnings)
    """
    target_mask_one_dim = np.zeros(expected_content_position_count_across_all_chunks, dtype=int)
    derived_target_words_in_original_case_list = (
        _compute_target_words_as_case_insensitive_set_difference_of_enhanced_minus_global(
            section_descriptor
        )
    )
    target_words_not_found_in_base = []
    for one_target_word_in_original_case in derived_target_words_in_original_case_list:
        word_stripped = one_target_word_in_original_case.strip()
        if not word_stripped:
            continue
        # Tokenize standalone using this stream's per-stream tokenizer's
        # tokenize_with_weights so behavior matches the base-prompt
        # tokenization that produced base_content_token_ids_flat_list.
        word_tokens_per_chunk = per_stream_tokenizer.tokenize_with_weights(word_stripped)
        word_content_ids = (
            _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad(
                word_tokens_per_chunk, end_token_id_for_this_stream
            )
        )
        if not word_content_ids:
            continue
        word_match_start_positions = _find_all_sublist_match_start_positions_within_superlist(
            base_content_token_ids_flat_list, word_content_ids
        )
        if not word_match_start_positions:
            target_words_not_found_in_base.append(word_stripped)
            continue
        word_length_in_tokens = len(word_content_ids)
        for one_match_start_position in word_match_start_positions:
            one_match_end_exclusive = one_match_start_position + word_length_in_tokens
            if one_match_end_exclusive > expected_content_position_count_across_all_chunks:
                continue
            target_mask_one_dim[one_match_start_position : one_match_end_exclusive] = 1
    return target_mask_one_dim, target_words_not_found_in_base


def _build_per_section_region_mask_over_content_positions_one_stream(
    section_descriptor,
    base_content_token_ids_flat_list,
    expected_content_position_count_across_all_chunks,
    per_stream_tokenizer,
    end_token_id_for_this_stream,
):
    """
    Region mask = positions in the base prompt where this section's
    enhanced_text appears as a token sublist. For a true region, the
    enhanced_text was inserted into the base prompt as a `(text:weight)`
    fragment so it WILL appear there contiguously.
    """
    region_mask_one_dim = np.zeros(expected_content_position_count_across_all_chunks, dtype=int)
    enhanced_text_tokens_per_chunk = per_stream_tokenizer.tokenize_with_weights(
        section_descriptor["enhanced_text"]
    )
    enhanced_content_ids = (
        _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad(
            enhanced_text_tokens_per_chunk, end_token_id_for_this_stream
        )
    )
    if not enhanced_content_ids:
        return region_mask_one_dim
    match_start_positions = _find_all_sublist_match_start_positions_within_superlist(
        base_content_token_ids_flat_list, enhanced_content_ids
    )
    enhanced_length_in_tokens = len(enhanced_content_ids)
    for one_match_start_position in match_start_positions:
        one_match_end_exclusive = one_match_start_position + enhanced_length_in_tokens
        if one_match_end_exclusive > expected_content_position_count_across_all_chunks:
            continue
        region_mask_one_dim[one_match_start_position : one_match_end_exclusive] = 1
    return region_mask_one_dim


def _encode_v3_for_one_stream_returning_final_embedding_and_pooled(
    clip_object,
    active_section_descriptors_list,
    stream_key_l_or_g,
):
    """
    v3 cutoff-style per-stream encoder. Builds the base prompt, performs
    cutoff masking math for true regions only, returns the final
    per-token embedding tensor + pooled output for this stream.

    Per-section text transforms (A1111 rewrite, orphan filter, etc.)
    should already have been applied to each descriptor's text fields
    BEFORE this is called.
    """
    per_stream_encoder_model, per_stream_tokenizer = (
        _get_per_stream_clip_text_encoder_model_and_tokenizer(clip_object, stream_key_l_or_g)
    )
    end_token_id_for_this_stream = per_stream_tokenizer.end_token

    base_prompt_text_for_this_stream, _per_section_fragment_or_none_list = (
        _build_per_stream_base_prompt_text_and_per_section_base_fragment_list(
            active_section_descriptors_list, stream_key_l_or_g
        )
    )

    # Tokenize base prompt for this stream.
    base_tokens_per_chunk_list = per_stream_tokenizer.tokenize_with_weights(
        base_prompt_text_for_this_stream
    )

    # Reshape any mismatched embedding tensors so the encoder accepts them.
    expected_last_dim_for_this_stream = getattr(per_stream_tokenizer, "embedding_size", None)
    if expected_last_dim_for_this_stream is not None:
        _reshape_any_mismatched_embedding_tensors_in_chunks_list_in_place_to_match_stream_expected_dim(
            base_tokens_per_chunk_list,
            expected_last_dim_for_this_stream,
            stream_key_l_or_g,
        )

    base_embedding_tensor, base_pooled_tensor_or_none = (
        per_stream_encoder_model.encode_token_weights(base_tokens_per_chunk_list)
    )

    # Determine true-region sections that ARE eligible on this stream.
    eligible_true_region_section_descriptors = [
        section_descriptor
        for section_descriptor in active_section_descriptors_list
        if section_descriptor["is_true_region"]
        and _select_per_stream_strength_value_for_section(section_descriptor, stream_key_l_or_g) != 0
    ]
    if not eligible_true_region_section_descriptors:
        return base_embedding_tensor, base_pooled_tensor_or_none

    number_of_chunks_after_tokenization = len(base_tokens_per_chunk_list)
    expected_content_position_count_across_all_chunks = (
        number_of_chunks_after_tokenization * CHUNK_CONTENT_TOKEN_COUNT_EXCLUDING_MARKERS
    )
    base_content_token_ids_flat_list = (
        _flatten_chunked_token_weight_pairs_into_content_token_ids_skipping_start_end_pad(
            base_tokens_per_chunk_list, end_token_id_for_this_stream
        )
    )

    per_section_target_masks_list = []
    per_section_region_masks_list = []
    per_section_effective_region_weights_list = []

    for section_descriptor in eligible_true_region_section_descriptors:
        section_target_mask_one_dim, _missing_words = (
            _build_per_section_target_mask_over_content_positions_one_stream(
                section_descriptor,
                base_content_token_ids_flat_list,
                expected_content_position_count_across_all_chunks,
                per_stream_tokenizer,
                end_token_id_for_this_stream,
                stream_key_l_or_g,
                clip_object,
            )
        )
        section_region_mask_one_dim = (
            _build_per_section_region_mask_over_content_positions_one_stream(
                section_descriptor,
                base_content_token_ids_flat_list,
                expected_content_position_count_across_all_chunks,
                per_stream_tokenizer,
                end_token_id_for_this_stream,
            )
        )
        if not section_target_mask_one_dim.any() or not section_region_mask_one_dim.any():
            # No target words or no region span found in base — skip this
            # section (validator widget surfaces the warning separately).
            continue
        per_section_target_masks_list.append(section_target_mask_one_dim)
        per_section_region_masks_list.append(section_region_mask_one_dim)
        per_section_strength_value = _select_per_stream_strength_value_for_section(
            section_descriptor, stream_key_l_or_g
        )
        per_section_effective_region_weights_list.append(
            section_descriptor["global_text_weight"] * per_section_strength_value
        )

    if not per_section_target_masks_list:
        return base_embedding_tensor, base_pooled_tensor_or_none

    global_target_mask_over_content = np.maximum.reduce(per_section_target_masks_list)
    global_region_mask_over_content = np.maximum.reduce(per_section_region_masks_list).astype(float)
    regions_overlap_count_sum = np.sum(np.stack(per_section_region_masks_list), axis=0).astype(float)
    regions_normalized_by_overlap_count = np.divide(
        1.0,
        regions_overlap_count_sum,
        out=np.zeros_like(regions_overlap_count_sum),
        where=regions_overlap_count_sum != 0,
    )

    mask_token_id_to_use = _resolve_mask_token_id_from_user_string_with_default(
        per_stream_tokenizer,
        HARDCODED_MASK_TOKEN_STRING_USE_END_OF_TEXT_DEFAULT,
        end_token_id_for_this_stream,
    )

    base_masked_tokens_per_chunk = (
        _build_masked_tokens_per_chunk_by_replacing_target_positions_with_mask_token_id(
            base_tokens_per_chunk_list,
            global_target_mask_over_content,
            mask_token_id_to_use,
            end_token_id_for_this_stream,
        )
    )
    base_masked_embedding_tensor, _unused_masked_pooled = (
        per_stream_encoder_model.encode_token_weights(base_masked_tokens_per_chunk)
    )

    # Hardcoded start_from_masked=1.0 and strict_mask=1.0
    base_start_embedding_tensor = base_masked_embedding_tensor
    base_outer_embedding_tensor = base_masked_embedding_tensor

    global_region_mask_over_full_chunks_one_dim = (
        _expand_content_position_mask_into_full_chunk_positions_with_start_end_padding(
            global_region_mask_over_content, number_of_chunks_after_tokenization
        )
    )

    region_contributions_summed_tensor = torch.zeros_like(base_embedding_tensor)
    for per_section_target_mask, per_section_region_mask, per_section_effective_region_weight in zip(
        per_section_target_masks_list,
        per_section_region_masks_list,
        per_section_effective_region_weights_list,
    ):
        per_section_mask_to_apply_one_dim = global_target_mask_over_content - per_section_target_mask
        per_section_masked_tokens_per_chunk = (
            _build_masked_tokens_per_chunk_by_replacing_target_positions_with_mask_token_id(
                base_tokens_per_chunk_list,
                per_section_mask_to_apply_one_dim,
                mask_token_id_to_use,
                end_token_id_for_this_stream,
            )
        )
        per_section_embedding_tensor, _unused_section_pooled = (
            per_stream_encoder_model.encode_token_weights(per_section_masked_tokens_per_chunk)
        )
        per_section_diff_from_base_start = per_section_embedding_tensor - base_start_embedding_tensor

        per_section_weight_per_content_position = (
            regions_normalized_by_overlap_count
            * per_section_region_mask
            * per_section_effective_region_weight
        )
        per_section_weight_per_full_chunk_position_one_dim = (
            _expand_content_position_mask_into_full_chunk_positions_with_start_end_padding(
                per_section_weight_per_content_position, number_of_chunks_after_tokenization
            )
        )
        per_section_weight_tensor_broadcastable = torch.tensor(
            per_section_weight_per_full_chunk_position_one_dim,
            dtype=base_embedding_tensor.dtype,
            device=base_embedding_tensor.device,
        ).unsqueeze(0).unsqueeze(-1)
        region_contributions_summed_tensor = (
            region_contributions_summed_tensor
            + per_section_diff_from_base_start * per_section_weight_tensor_broadcastable
        )

    global_region_mask_tensor_broadcastable = torch.tensor(
        global_region_mask_over_full_chunks_one_dim,
        dtype=base_embedding_tensor.dtype,
        device=base_embedding_tensor.device,
    ).unsqueeze(0).unsqueeze(-1)
    final_embedding_tensor = (
        base_start_embedding_tensor * global_region_mask_tensor_broadcastable
        + base_outer_embedding_tensor * (1.0 - global_region_mask_tensor_broadcastable)
        + region_contributions_summed_tensor
    )

    return final_embedding_tensor, base_pooled_tensor_or_none


def _encode_active_v3_sections_into_one_sdxl_conditioning_entry(
    clip_object,
    active_section_descriptors_list,
):
    """
    Top-level per-streams driver. Encodes L and G independently using
    v3 cutoff math, then SDXL-combines into one [tokens_tensor, metadata]
    entry. Pooled output taken from G stream.
    """
    final_embedding_tensor_for_l_stream, _l_pooled_unused = (
        _encode_v3_for_one_stream_returning_final_embedding_and_pooled(
            clip_object, active_section_descriptors_list, "l"
        )
    )
    final_embedding_tensor_for_g_stream, g_pooled_output_tensor_or_none = (
        _encode_v3_for_one_stream_returning_final_embedding_and_pooled(
            clip_object, active_section_descriptors_list, "g"
        )
    )

    cut_to_shared_sequence_length = min(
        final_embedding_tensor_for_l_stream.shape[1],
        final_embedding_tensor_for_g_stream.shape[1],
    )
    sdxl_combined_tokens_tensor = torch.cat(
        [
            final_embedding_tensor_for_l_stream[:, :cut_to_shared_sequence_length],
            final_embedding_tensor_for_g_stream[:, :cut_to_shared_sequence_length],
        ],
        dim=-1,
    )

    entry_metadata_dict = {}
    if g_pooled_output_tensor_or_none is not None:
        entry_metadata_dict["pooled_output"] = g_pooled_output_tensor_or_none

    return [sdxl_combined_tokens_tensor, entry_metadata_dict]


# -------------- SDXL size/crop metadata --------------

def _clamp_numeric_value_inclusive(value_to_clamp, minimum_allowed, maximum_allowed):
    return max(minimum_allowed, min(maximum_allowed, value_to_clamp))


def _compute_sdxl_size_and_crop_metadata_fields(
    target_image_width, target_image_height, zoom_factor, offset_x_clamped, offset_y_clamped
):
    source_image_width = int(round(target_image_width * zoom_factor))
    source_image_height = int(round(target_image_height * zoom_factor))
    maximum_horizontal_crop_offset = max(0, source_image_width - target_image_width)
    maximum_vertical_crop_offset = max(0, source_image_height - target_image_height)
    crop_w_value = int(round(((offset_x_clamped + 1.0) * 0.5) * maximum_horizontal_crop_offset))
    crop_h_value = int(round(((offset_y_clamped + 1.0) * 0.5) * maximum_vertical_crop_offset))
    return {
        "width": source_image_width,
        "height": source_image_height,
        "crop_w": crop_w_value,
        "crop_h": crop_h_value,
        "target_width": target_image_width,
        "target_height": target_image_height,
        "original_size_as_tuple": (source_image_width, source_image_height),
        "crop_coords_top_left": (crop_w_value, crop_h_value),
        "target_size_as_tuple": (target_image_width, target_image_height),
    }


def _resolve_target_image_width_and_height_from_optional_latent_or_defaults(latent_or_none):
    if latent_or_none is None:
        return (DEFAULT_LATENT_IMAGE_WIDTH_WHEN_NO_LATENT_INPUT, DEFAULT_LATENT_IMAGE_HEIGHT_WHEN_NO_LATENT_INPUT)
    latent_samples_tensor = latent_or_none.get("samples")
    if latent_samples_tensor is None:
        return (DEFAULT_LATENT_IMAGE_WIDTH_WHEN_NO_LATENT_INPUT, DEFAULT_LATENT_IMAGE_HEIGHT_WHEN_NO_LATENT_INPUT)
    target_image_width = latent_samples_tensor.shape[3] * LATENT_TO_IMAGE_SPATIAL_SCALE_FACTOR
    target_image_height = latent_samples_tensor.shape[2] * LATENT_TO_IMAGE_SPATIAL_SCALE_FACTOR
    return (target_image_width, target_image_height)


# -------------- the node class --------------

class CLIPTextEncodeSDXLV3GlobalAndEnhanced:
    @classmethod
    def INPUT_TYPES(cls):
        required_inputs_dict = {
            "clip": ("CLIP",),
            "upscaled_conditioning_multiplier": ("FLOAT", {
                "default": UPSCALED_CONDITIONING_MULTIPLIER_DEFAULT_VALUE,
                "min": UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE,
                "step": 0.01,
            }),
            "section_count": ("INT", {
                "default": DEFAULT_SECTION_COUNT_VALUE,
                "min": 0,
                "max": MAX_SECTION_COUNT_SUPPORTED,
                "step": 1,
            }),
            "support_a1111_style_embedding_text": ("BOOLEAN", {"default": True}),
            "remove_text_for_unsupported_embeddings": ("BOOLEAN", {"default": True}),
            "filter_known_a1111_embedding_tags_not_installed_locally": ("BOOLEAN", {
                "default": True,
                "tooltip": (
                    "List can be modified in custom node folder: "
                    "known_a1111_embedding_names_to_filter_when_not_installed_locally.txt"
                ),
            }),
            # Zoom-effect group (zoom + offset_x + offset_y). A canvas-drawn
            # header label is inserted above these by the dedicated frontend
            # extension web/clip_text_encode_sdxl_v3_global_and_enhanced.js.
            "zoom": ("FLOAT", {
                "default": ZOOM_DEFAULT_VALUE,
                "min": ZOOM_MINIMUM_VALUE,
                "max": ZOOM_MAXIMUM_VALUE,
                "step": 0.01,
            }),
            "offset_x": ("FLOAT", {
                "default": OFFSET_DEFAULT_VALUE,
                "min": OFFSET_MINIMUM_VALUE,
                "max": OFFSET_MAXIMUM_VALUE,
                "step": 0.01,
            }),
            "offset_y": ("FLOAT", {
                "default": OFFSET_DEFAULT_VALUE,
                "min": OFFSET_MINIMUM_VALUE,
                "max": OFFSET_MAXIMUM_VALUE,
                "step": 0.01,
            }),
        }
        for section_index_for_declaration in range(1, MAX_SECTION_COUNT_SUPPORTED + 1):
            required_inputs_dict[f"section_{section_index_for_declaration}_global_text"] = (
                "STRING", {"multiline": True, "default": ""},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_enhanced_text"] = (
                "STRING", {"multiline": True, "default": ""},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_global_text_weight"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_enhanced_text_weight"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_clip_l_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_clip_g_strength"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
        return {
            "required": required_inputs_dict,
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "upscaled_conditioning", "reference_full_prompt")
    FUNCTION = "encode_v3_global_and_enhanced"
    CATEGORY = "unified-conditioning-merge"

    def encode_v3_global_and_enhanced(
        self,
        clip,
        upscaled_conditioning_multiplier,
        section_count,
        support_a1111_style_embedding_text,
        remove_text_for_unsupported_embeddings,
        filter_known_a1111_embedding_tags_not_installed_locally,
        zoom,
        offset_x,
        offset_y,
        latent=None,
        **kwargs_for_individual_section_widget_values,
    ):
        active_section_descriptors_list = (
            _collect_active_v3_section_descriptors_from_kwargs_in_declaration_order(
                kwargs_for_individual_section_widget_values, section_count
            )
        )

        # Per-section text transforms — applied to BOTH global_text and
        # enhanced_text per section, in v1's order (orphan filter, A1111
        # rewrite, shape-mismatch warnings, unsupported strip). The
        # orphan filter uses the curated text-file-based list ONLY
        # (file: known_a1111_embedding_names_to_filter_when_not_installed_locally.txt
        # at the plugin root). No per-node custom override widget — matches v1.
        for section_descriptor_to_transform in active_section_descriptors_list:
            section_descriptor_to_transform["global_text"] = (
                _apply_v3_per_text_transforms_to_one_text_string(
                    section_descriptor_to_transform["global_text"],
                    clip,
                    bool(support_a1111_style_embedding_text),
                    bool(remove_text_for_unsupported_embeddings),
                    bool(filter_known_a1111_embedding_tags_not_installed_locally),
                )
            )
            section_descriptor_to_transform["enhanced_text"] = (
                _apply_v3_per_text_transforms_to_one_text_string(
                    section_descriptor_to_transform["enhanced_text"],
                    clip,
                    bool(support_a1111_style_embedding_text),
                    bool(remove_text_for_unsupported_embeddings),
                    bool(filter_known_a1111_embedding_tags_not_installed_locally),
                )
            )
            # Re-evaluate true-region after text transforms (one or both
            # texts may have been emptied by the strip passes).
            normalized_global = (section_descriptor_to_transform["global_text"] or "").lower()
            normalized_enhanced = (section_descriptor_to_transform["enhanced_text"] or "").lower()
            section_descriptor_to_transform["is_true_region"] = (
                bool(normalized_enhanced) and normalized_enhanced != normalized_global
            )
        # Drop sections that became fully empty after transforms.
        active_section_descriptors_list = [
            section_descriptor for section_descriptor in active_section_descriptors_list
            if section_descriptor["global_text"] or section_descriptor["enhanced_text"]
        ]

        # Resolve SDXL geometry (primary + upscaled), identical to v1.
        primary_target_image_width, primary_target_image_height = (
            _resolve_target_image_width_and_height_from_optional_latent_or_defaults(latent)
        )
        conditioning_upscale_factor_clamped = max(
            UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE, float(upscaled_conditioning_multiplier)
        )
        upscaled_target_image_width = int(round(primary_target_image_width * conditioning_upscale_factor_clamped))
        upscaled_target_image_height = int(round(primary_target_image_height * conditioning_upscale_factor_clamped))

        zoom_factor_clamped = max(ZOOM_MINIMUM_VALUE, float(zoom))
        offset_x_clamped = _clamp_numeric_value_inclusive(
            float(offset_x), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE
        )
        offset_y_clamped = _clamp_numeric_value_inclusive(
            float(offset_y), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE
        )
        primary_sdxl_size_and_crop_metadata_fields = _compute_sdxl_size_and_crop_metadata_fields(
            primary_target_image_width, primary_target_image_height,
            zoom_factor_clamped, offset_x_clamped, offset_y_clamped,
        )
        upscaled_targets_differ_from_primary = (
            upscaled_target_image_width != primary_target_image_width
            or upscaled_target_image_height != primary_target_image_height
        )
        if upscaled_targets_differ_from_primary:
            upscaled_sdxl_size_and_crop_metadata_fields = _compute_sdxl_size_and_crop_metadata_fields(
                upscaled_target_image_width, upscaled_target_image_height,
                zoom_factor_clamped, offset_x_clamped, offset_y_clamped,
            )
        else:
            upscaled_sdxl_size_and_crop_metadata_fields = primary_sdxl_size_and_crop_metadata_fields

        # Empty-prompt fallback: no active sections at all.
        if not active_section_descriptors_list:
            empty_tokens_dict = clip.tokenize("")
            primary_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=primary_sdxl_size_and_crop_metadata_fields,
            )
            upscaled_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=upscaled_sdxl_size_and_crop_metadata_fields,
            )
            return (primary_empty_conditioning_list, upscaled_empty_conditioning_list, "")

        # Encode via v3 cutoff math (single composite entry).
        try:
            raw_conditioning_entry_from_v3_encoder = (
                _encode_active_v3_sections_into_one_sdxl_conditioning_entry(
                    clip, active_section_descriptors_list
                )
            )
        except Exception as v3_encoding_failure:
            logging.warning(
                f"CLIPTextEncodeSDXLV3GlobalAndEnhanced: encoding failed "
                f"({type(v3_encoding_failure).__name__}: {v3_encoding_failure}). "
                f"Falling back to empty conditioning."
            )
            empty_tokens_dict = clip.tokenize("")
            primary_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=primary_sdxl_size_and_crop_metadata_fields,
            )
            upscaled_empty_conditioning_list = clip.encode_from_tokens_scheduled(
                empty_tokens_dict, add_dict=upscaled_sdxl_size_and_crop_metadata_fields,
            )
            return (primary_empty_conditioning_list, upscaled_empty_conditioning_list, "")

        # Stamp both primary and upscaled SDXL metadata onto the same
        # token tensor (shared by reference; only metadata dict differs).
        raw_tokens_tensor = raw_conditioning_entry_from_v3_encoder[0]
        raw_metadata_dict = raw_conditioning_entry_from_v3_encoder[1]

        primary_entry_metadata_dict = dict(raw_metadata_dict)
        primary_entry_metadata_dict.update(primary_sdxl_size_and_crop_metadata_fields)
        upscaled_entry_metadata_dict = dict(raw_metadata_dict)
        upscaled_entry_metadata_dict.update(upscaled_sdxl_size_and_crop_metadata_fields)

        # Reference prompt = the "visible base prompt" the model actually
        # encoded (comma-join of each section's enhanced_text if true-region,
        # else global_text). Both streams produce the same base text since
        # weights only affect the (text:weight) wrapping; show the G-stream
        # build for canonical display.
        reference_base_prompt_text_for_g_stream, _ = (
            _build_per_stream_base_prompt_text_and_per_section_base_fragment_list(
                active_section_descriptors_list, "g"
            )
        )

        return (
            [[raw_tokens_tensor, primary_entry_metadata_dict]],
            [[raw_tokens_tensor, upscaled_entry_metadata_dict]],
            reference_base_prompt_text_for_g_stream,
        )


# ──────────────────────────────────────────────────────────────────────
# Public aliases for shared use by the chain-based detail-isolation node
# pair (CLIPTextEncodeSDXLEnhancedDetailIsolation + Section). These are
# explicit re-exports of v3's encoding helpers under non-underscore
# names so sibling nodes can import them without reaching into
# private-prefixed symbols. Same implementation, public-facing names.
#
# This is an intentional, documented shared-API contract. v3's internal
# call sites continue to use the underscore names. Sibling nodes import
# the public names below.
# ──────────────────────────────────────────────────────────────────────

apply_v3_per_text_transforms_to_one_text_string = _apply_v3_per_text_transforms_to_one_text_string
encode_active_v3_sections_into_one_sdxl_conditioning_entry = _encode_active_v3_sections_into_one_sdxl_conditioning_entry
build_per_stream_base_prompt_text_and_per_section_base_fragment_list = _build_per_stream_base_prompt_text_and_per_section_base_fragment_list
compute_sdxl_size_and_crop_metadata_fields = _compute_sdxl_size_and_crop_metadata_fields
resolve_target_image_width_and_height_from_optional_latent_or_defaults = _resolve_target_image_width_and_height_from_optional_latent_or_defaults
clamp_numeric_value_inclusive = _clamp_numeric_value_inclusive
