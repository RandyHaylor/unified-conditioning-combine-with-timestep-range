"""
CLIP Text Encode (Cutoff Region Separation)

Single-node prompt builder using a self-contained, MIT-licensed Cutoff-style
algorithm (see cutoff_per_stream_isolation.py) that runs INDEPENDENTLY per
CLIP stream. This is what lets per-section CLIP routing actually work
correctly for SDXL: when a section is marked "L only", the L stream encodes
that section's text and the G stream encodes the empty prompt (giving
natural CLIP-G empty embeddings, NOT zero-masked tensors that would be
out-of-distribution for the model).

For each section, the dropdown chooses which stream(s) this section
contributes to:
    L+G   — section text goes to BOTH text_l and text_g (typical)
    L     — section text goes to text_l only; text_g for this section's group
            is the empty string (CLIP-G encodes the natural empty prompt)
    G     — symmetric for G

Sections sharing a clip choice are GROUPED. Per group, we build the
per-stream prompts and run cutoff-style isolation independently on each
stream. The L-stream and G-stream results are concatenated along the last
dim to form the SDXL combined tensor (768 + 1280 = 2048).

All groups' resulting CONDITIONING entries are emitted together as a
multi-entry CONDITIONING (combine-style — sampler treats each group as a
parallel branch).

SDXL size/crop metadata (zoom + offsets, target W/H from optional LATENT
defaulting to 1024x1024) is stamped onto every output entry.

NO external plugin dependency. Self-contained.
"""

import inspect
import logging
import os
import re
import sys

import torch

import folder_paths

from .cutoff_per_stream_isolation import encode_one_stream_with_cutoff_style_region_isolation


MAX_SECTION_COUNT_SUPPORTED = 16
DEFAULT_SECTION_COUNT_VALUE = 3

CLIP_STREAM_PASS_BOTH_L_AND_G = "Pass L+G"
CLIP_STREAM_PASS_L_ONLY = "Pass L"
CLIP_STREAM_PASS_G_ONLY = "Pass G"
CLIP_STREAM_PASS_CLASSIC_UPSTREAM_CUTOFF = "Classic"
CLIP_STREAM_PASS_CHOICES_IN_DROPDOWN_ORDER = [
    CLIP_STREAM_PASS_BOTH_L_AND_G,
    CLIP_STREAM_PASS_L_ONLY,
    CLIP_STREAM_PASS_G_ONLY,
    CLIP_STREAM_PASS_CLASSIC_UPSTREAM_CUTOFF,
]

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


# -------------- Classic mode: upstream ComfyUI_Cutoff plugin lookup --------------

def _find_loaded_upstream_cutoff_module_in_sys_modules_or_none():
    """
    Locate the ComfyUI_Cutoff plugin's `cutoff` module by scanning sys.modules
    for one that exposes the expected interface AS A REAL Python class +
    callable function (to avoid matching torch._OpNamespace and friends whose
    __getattr__ returns truthy for any name). Prefer modules whose name
    contains "cutoff" (case-insensitive). Used ONLY when a section is marked
    with the Classic clip_pass_choice — other modes use our self-contained
    per-stream implementation.
    """
    matching_candidate_modules_keyed_by_name = {}
    for module_name_in_sys_modules, loaded_module_object in list(sys.modules.items()):
        if loaded_module_object is None:
            continue
        try:
            finalize_attribute_or_none = getattr(loaded_module_object, "finalize_clip_regions", None)
            base_prompt_attribute_or_none = getattr(loaded_module_object, "CLIPRegionsBasePrompt", None)
        except Exception:
            continue
        if finalize_attribute_or_none is None or base_prompt_attribute_or_none is None:
            continue
        if not callable(finalize_attribute_or_none) or not inspect.isclass(base_prompt_attribute_or_none):
            continue
        matching_candidate_modules_keyed_by_name[module_name_in_sys_modules] = loaded_module_object
    if not matching_candidate_modules_keyed_by_name:
        return None
    for candidate_module_name, candidate_module in matching_candidate_modules_keyed_by_name.items():
        if "cutoff" in candidate_module_name.lower():
            return candidate_module
    return next(iter(matching_candidate_modules_keyed_by_name.values()))


def _encode_one_group_via_classic_upstream_cutoff_plugin(
    clip_object,
    section_descriptors_in_classic_group_list,
    join_separator_string,
    mask_token_string,
    strict_mask_value,
    start_from_masked_value,
):
    """
    Runs the upstream ComfyUI_Cutoff plugin directly for a group of sections
    marked Classic. Mathematically identical to using the upstream plugin
    standalone — for A/B comparison against our per-stream implementation.

    Raises RuntimeError if the upstream plugin isn't installed.
    """
    upstream_cutoff_module = _find_loaded_upstream_cutoff_module_in_sys_modules_or_none()
    if upstream_cutoff_module is None:
        raise RuntimeError(
            "CLIPTextEncodeWithCutoffRegionSeparation: Classic clip_pass_choice "
            "requires the ComfyUI_Cutoff plugin to be installed (it's used as "
            "the reference implementation for A/B comparison). Install from "
            "https://github.com/BlenderNeko/ComfyUI_Cutoff into ComfyUI/custom_nodes/ "
            "and restart ComfyUI. Other clip_pass_choice options (Pass L+G / Pass L / "
            "Pass G) don't need it — they use our self-contained per-stream code."
        )

    classic_group_full_prompt_text = _build_full_prompt_text_for_one_group_of_sections(
        section_descriptors_in_classic_group_list, join_separator_string
    )

    base_prompt_node_instance = upstream_cutoff_module.CLIPRegionsBasePrompt()
    base_state_tuple_from_init_prompt = base_prompt_node_instance.init_prompt(
        clip_object, classic_group_full_prompt_text
    )
    current_clip_regions_state = base_state_tuple_from_init_prompt[0]

    add_region_node_instance = upstream_cutoff_module.CLIPSetRegion()
    for section_descriptor_in_classic_group in section_descriptors_in_classic_group_list:
        if not section_descriptor_in_classic_group["isolate"]:
            continue
        try:
            next_state_tuple_after_region_add = add_region_node_instance.add_clip_region(
                current_clip_regions_state,
                section_descriptor_in_classic_group["text"],
                section_descriptor_in_classic_group["text"],
                section_descriptor_in_classic_group["weight"],
            )
            current_clip_regions_state = next_state_tuple_after_region_add[0]
        except Exception as upstream_cutoff_register_region_exception:
            logging.warning(
                "CLIPTextEncodeWithCutoffRegionSeparation (Classic): skipped isolate "
                f"registration for section text {section_descriptor_in_classic_group['text']!r} "
                f"(upstream Cutoff raised {type(upstream_cutoff_register_region_exception).__name__}: "
                f"{upstream_cutoff_register_region_exception})."
            )

    finalize_return_tuple_from_upstream_cutoff = upstream_cutoff_module.finalize_clip_regions(
        current_clip_regions_state,
        mask_token_string,
        float(strict_mask_value),
        float(start_from_masked_value),
    )
    # Upstream returns a CONDITIONING list. For Classic we just take that list
    # as-is and let the caller stamp SDXL size/crop metadata onto each entry.
    return finalize_return_tuple_from_upstream_cutoff[0]


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


def _collect_active_non_empty_sections_from_kwargs(kwargs_dict, active_section_count):
    active_section_descriptors_list = []
    for section_index in range(1, int(active_section_count) + 1):
        section_text_raw = kwargs_dict.get(f"section_{section_index}_text", "")
        section_isolate_raw = kwargs_dict.get(f"section_{section_index}_isolate", True)
        section_weight_raw = kwargs_dict.get(f"section_{section_index}_weight", 1.0)
        section_clip_pass_choice_raw = kwargs_dict.get(
            f"section_{section_index}_clip", CLIP_STREAM_PASS_BOTH_L_AND_G
        )
        section_text_normalized_whitespace = WHITESPACE_RUN_REGEX_FOR_NORMALIZING_SECTION_TEXT.sub(
            " ", (section_text_raw or "")
        ).strip()
        if not section_text_normalized_whitespace:
            continue
        active_section_descriptors_list.append({
            "text": section_text_normalized_whitespace,
            "isolate": bool(section_isolate_raw),
            "weight": float(section_weight_raw),
            "clip_pass_choice": str(section_clip_pass_choice_raw),
        })
    return active_section_descriptors_list


def _build_effective_per_section_prompt_text_with_optional_clip_attention_weight_wrap(section_descriptor):
    section_bare_text = section_descriptor["text"]
    if section_descriptor["isolate"]:
        return section_bare_text
    section_weight_value = section_descriptor["weight"]
    if abs(section_weight_value - 1.0) < 1e-9:
        return section_bare_text
    return f"({section_bare_text}:{section_weight_value:.3f})"


def _build_full_prompt_text_for_one_group_of_sections(section_descriptors_in_group_list, join_separator_string):
    per_section_effective_prompt_text_fragments = [
        _build_effective_per_section_prompt_text_with_optional_clip_attention_weight_wrap(d)
        for d in section_descriptors_in_group_list
    ]
    return join_separator_string.join(per_section_effective_prompt_text_fragments)


def _group_active_sections_by_their_clip_pass_choice(active_section_descriptors_list):
    grouped_sections_by_clip_pass_choice = {}
    for section_descriptor in active_section_descriptors_list:
        clip_pass_choice_for_this_section = section_descriptor["clip_pass_choice"]
        if clip_pass_choice_for_this_section not in grouped_sections_by_clip_pass_choice:
            grouped_sections_by_clip_pass_choice[clip_pass_choice_for_this_section] = []
        grouped_sections_by_clip_pass_choice[clip_pass_choice_for_this_section].append(section_descriptor)
    return grouped_sections_by_clip_pass_choice


# -------------- per-group encoding using per-stream cutoff --------------

def _build_isolate_target_text_and_weight_pairs_for_one_group(section_descriptors_in_group_list):
    """
    Returns [{"target_text": str, "weight": float}, ...] for sections in the
    group that have isolate=True. Non-isolate sections are skipped (their
    weight is applied via the inline `(text:weight)` wrap in the group's
    joined prompt text).
    """
    isolate_target_text_and_weight_pairs_list = []
    for section_descriptor in section_descriptors_in_group_list:
        if not section_descriptor["isolate"]:
            continue
        isolate_target_text_and_weight_pairs_list.append({
            "target_text": section_descriptor["text"],
            "weight": section_descriptor["weight"],
        })
    return isolate_target_text_and_weight_pairs_list


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
    `embedding:...`) that matches a name in the known-A1111-embedding-names
    list (loaded from known_a1111_embedding_names_to_filter_when_not_installed_locally.txt)
    AND does NOT correspond to a file installed in the user's embeddings
    folder, drops the tag from the prompt entirely. Logs each removal.
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
    """Tokenizes `embedding:NAME` in isolation and checks per stream whether
    the loaded embedding tensor's last dim matches that stream's expected
    embedding_size. Returns True if ANY stream mismatches."""
    try:
        isolated_tokenization_per_stream = clip_object.tokenize(f"embedding:{embedding_name_string}")
    except Exception:
        return False
    if not isinstance(isolated_tokenization_per_stream, dict):
        return False
    top_level_tokenizer_object_or_none = getattr(clip_object, "tokenizer", None)
    for stream_name_key, chunks_list_for_this_stream in isolated_tokenization_per_stream.items():
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


def _encode_one_group_via_plain_stock_clip_without_any_isolation(
    clip_object,
    section_descriptors_in_group_list,
    clip_pass_choice_for_this_group,
    join_separator_string,
):
    """
    Stock-style fallback used when the cutoff isolation path fails (e.g.,
    prompt contains an embedding with shape that doesn't match the target
    CLIP stream's embedding dim). Returns a CONDITIONING list (typically
    one entry).

    For Pass L+G: same text fed to both streams.
    For Pass L only: text fed to L; G is the empty prompt.
    For Pass G only: symmetric.
    For Classic (which has its own encoder path): also handled as L+G here
    since we can't reproduce the classic plugin's behavior in fallback.
    """
    group_full_prompt_text = _build_full_prompt_text_for_one_group_of_sections(
        section_descriptors_in_group_list, join_separator_string
    )

    if clip_pass_choice_for_this_group == CLIP_STREAM_PASS_L_ONLY:
        text_for_l_stream = group_full_prompt_text
        text_for_g_stream = ""
    elif clip_pass_choice_for_this_group == CLIP_STREAM_PASS_G_ONLY:
        text_for_l_stream = ""
        text_for_g_stream = group_full_prompt_text
    else:
        text_for_l_stream = group_full_prompt_text
        text_for_g_stream = group_full_prompt_text

    tokens_for_stock_fallback_encode = clip_object.tokenize(text_for_g_stream)
    tokens_for_stock_fallback_encode["l"] = clip_object.tokenize(text_for_l_stream)["l"]

    # Match stock CLIPTextEncodeSDXL's pad-shorter-side-with-empty logic.
    empty_tokens_dict_for_padding = clip_object.tokenize("")
    while len(tokens_for_stock_fallback_encode["l"]) < len(tokens_for_stock_fallback_encode["g"]):
        tokens_for_stock_fallback_encode["l"] += empty_tokens_dict_for_padding["l"]
    while len(tokens_for_stock_fallback_encode["l"]) > len(tokens_for_stock_fallback_encode["g"]):
        tokens_for_stock_fallback_encode["g"] += empty_tokens_dict_for_padding["g"]

    return clip_object.encode_from_tokens_scheduled(tokens_for_stock_fallback_encode)


def _encode_one_group_into_one_sdxl_conditioning_entry(
    clip_object,
    section_descriptors_in_group_list,
    clip_pass_choice_for_this_group,
    join_separator_string,
    mask_token_string,
    strict_mask_value,
    start_from_masked_value,
    support_a1111_style_embedding_text_setting=True,
    remove_text_for_unsupported_embeddings_setting=True,
    filter_known_a1111_embedding_tags_not_installed_locally_setting=True,
):
    """
    Returns a single CONDITIONING entry [combined_tokens_tensor, metadata_dict]
    for one group's combined prompt encoded with proper per-stream handling
    (no zero-masking — empty streams get natural CLIP empty-prompt
    encodings).
    """
    group_full_prompt_text_for_isolate_streams = _build_full_prompt_text_for_one_group_of_sections(
        section_descriptors_in_group_list, join_separator_string
    )
    # Orphan-embedding-tag filter: drop bare comma-separated tags that
    # match the curated known-A1111-embedding-names list (loaded from
    # known_a1111_embedding_names_to_filter_when_not_installed_locally.txt)
    # but are NOT installed in the user's embeddings directory. This
    # prevents other people's shared A1111 prompts (where bare tags like
    # `bad-hands-5` reference embedding files the current user doesn't
    # have) from being encoded as plain text and accidentally influencing
    # the image.
    if filter_known_a1111_embedding_tags_not_installed_locally_setting:
        available_embedding_lowercase_stem_to_filenames_map = (
            _build_lookup_of_available_embedding_stems_to_their_filename_list()
        )
        group_full_prompt_text_for_isolate_streams = (
            _strip_orphan_a1111_bare_tags_matching_known_names_list_but_not_installed_locally(
                group_full_prompt_text_for_isolate_streams,
                available_embedding_lowercase_stem_to_filenames_map,
            )
        )
    # A1111-style sweep: identify bare tags that exactly match an embedding
    # filename stem (case-insensitive) and rewrite them to ComfyUI's
    # `embedding:NAME` form so the tokenizer loads them. Logs each rewrite
    # and warns when multiple files share the same stem. User-toggleable.
    if support_a1111_style_embedding_text_setting:
        group_full_prompt_text_for_isolate_streams = (
            _rewrite_a1111_style_bare_embedding_tags_to_comfyui_embedding_prefix_form(
                group_full_prompt_text_for_isolate_streams
            )
        )
    # Pre-scan for embedding files whose dim doesn't match this model's
    # streams, so the user sees the FILE NAME of the offending embedding
    # (stock's own warning only prints dims, not the name).
    _warn_about_each_mismatched_shape_embedding_reference_in_prompt_text(
        group_full_prompt_text_for_isolate_streams, clip_object
    )
    # If the user has chosen to drop unsupported embeddings entirely, strip
    # every `embedding:NAME` reference whose file's dim won't match this
    # model's streams from the prompt text. The reference vanishes from the
    # prompt before tokenization — no tensor gets loaded, no pad/truncate
    # salvage runs, no influence on the image.
    if remove_text_for_unsupported_embeddings_setting:
        group_full_prompt_text_for_isolate_streams = (
            _strip_unsupported_embedding_references_from_prompt_text(
                group_full_prompt_text_for_isolate_streams, clip_object
            )
        )
    isolate_target_text_and_weight_pairs_for_this_group = _build_isolate_target_text_and_weight_pairs_for_one_group(
        section_descriptors_in_group_list
    )

    # Decide each stream's prompt based on the group's CLIP pass choice.
    if clip_pass_choice_for_this_group == CLIP_STREAM_PASS_BOTH_L_AND_G:
        prompt_text_for_l_stream = group_full_prompt_text_for_isolate_streams
        prompt_text_for_g_stream = group_full_prompt_text_for_isolate_streams
        run_isolation_on_l_stream = True
        run_isolation_on_g_stream = True
    elif clip_pass_choice_for_this_group == CLIP_STREAM_PASS_L_ONLY:
        prompt_text_for_l_stream = group_full_prompt_text_for_isolate_streams
        prompt_text_for_g_stream = ""
        run_isolation_on_l_stream = True
        run_isolation_on_g_stream = False
    elif clip_pass_choice_for_this_group == CLIP_STREAM_PASS_G_ONLY:
        prompt_text_for_l_stream = ""
        prompt_text_for_g_stream = group_full_prompt_text_for_isolate_streams
        run_isolation_on_l_stream = False
        run_isolation_on_g_stream = True
    else:
        raise ValueError(
            f"Unknown clip_pass_choice {clip_pass_choice_for_this_group!r}"
        )

    # Run cutoff-style isolation per stream. For the "empty" stream we pass
    # an empty isolate list so it just encodes the empty prompt normally.
    final_embedding_tensor_for_l_stream, _l_pooled_unused = encode_one_stream_with_cutoff_style_region_isolation(
        clip_object,
        "l",
        prompt_text_for_l_stream,
        isolate_target_text_and_weight_pairs_for_this_group if run_isolation_on_l_stream else [],
        mask_token_string,
        strict_mask_value,
        start_from_masked_value,
    )
    final_embedding_tensor_for_g_stream, g_pooled_output_tensor_or_none = encode_one_stream_with_cutoff_style_region_isolation(
        clip_object,
        "g",
        prompt_text_for_g_stream,
        isolate_target_text_and_weight_pairs_for_this_group if run_isolation_on_g_stream else [],
        mask_token_string,
        strict_mask_value,
        start_from_masked_value,
    )

    # SDXL combine: concat L (768) + G (1280) along last dim. Truncate to
    # shorter sequence length to handle chunk-count mismatches.
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

class CLIPTextEncodeWithCutoffRegionSeparation:
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
                "min": 1,
                "max": MAX_SECTION_COUNT_SUPPORTED,
                "step": 1,
            }),
            "join_separator": ("STRING", {"multiline": False, "default": ","}),
            "mask_token": ("STRING", {"multiline": False, "default": ""}),
            "strict_mask": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "start_from_masked": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
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
            # extension web/clip_text_encode_with_cutoff_region_separation.js.
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
            required_inputs_dict[f"section_{section_index_for_declaration}_text"] = (
                "STRING", {"multiline": True, "default": ""},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_isolate"] = (
                "BOOLEAN", {"default": True},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_weight"] = (
                "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
            )
            required_inputs_dict[f"section_{section_index_for_declaration}_clip"] = (
                CLIP_STREAM_PASS_CHOICES_IN_DROPDOWN_ORDER,
                {"default": CLIP_STREAM_PASS_BOTH_L_AND_G},
            )
        return {
            "required": required_inputs_dict,
            "optional": {
                "latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "upscaled_conditioning", "reference_full_prompt")
    FUNCTION = "encode_with_grouped_per_stream_cutoff_and_sdxl_zoom"
    CATEGORY = "unified-conditioning-merge"

    def encode_with_grouped_per_stream_cutoff_and_sdxl_zoom(
        self,
        clip,
        upscaled_conditioning_multiplier,
        section_count,
        join_separator,
        mask_token,
        strict_mask,
        start_from_masked,
        zoom,
        offset_x,
        offset_y,
        support_a1111_style_embedding_text=True,
        remove_text_for_unsupported_embeddings=True,
        filter_known_a1111_embedding_tags_not_installed_locally=True,
        latent=None,
        **kwargs_for_individual_section_widget_values,
    ):
        active_section_descriptors_list = _collect_active_non_empty_sections_from_kwargs(
            kwargs_for_individual_section_widget_values, section_count
        )

        # Stock CLIPTextEncode emits a "shape mismatch when trying to apply
        # embedding" WARNING when a referenced textual-inversion embedding's
        # saved hidden-dim does not match a CLIP stream's expected hidden-dim
        # (e.g. SD1.5 768-dim embedding used in an SDXL prompt). Stock then
        # ignores that embedding and still produces an image. Our per-stream
        # cutoff path sometimes does not surface this warning, which makes it
        # look like the user supplied a working prompt when they actually
        # didn't. Proactively scan every active section's text at entry and
        # emit the same warning per mismatched embedding occurrence, before
        # any encoding work runs.
        for one_section_descriptor_to_scan_for_mismatched_embeddings in active_section_descriptors_list:
            _emit_stock_style_shape_mismatch_warning_for_each_embedding_in_text_that_will_not_match_its_clip_stream_dim(
                clip,
                one_section_descriptor_to_scan_for_mismatched_embeddings["text"],
            )

        # Resolve target W/H for the PRIMARY conditioning's SDXL metadata.
        primary_target_image_width, primary_target_image_height = _resolve_target_image_width_and_height_from_optional_latent_or_defaults(latent)
        # The UPSCALED conditioning's target W/H is the primary target W/H
        # multiplied by the conditioning_upscale_by widget value. Computed
        # in Python (not from a downstream latent input) because the
        # downstream latent doesn't exist at prompt-validation time —
        # ComfyUI validates the entire graph upfront, before any node has
        # run, so we can't read an upscaled-by-a-later-stage latent here.
        conditioning_upscale_factor_clamped = max(
            UPSCALED_CONDITIONING_MULTIPLIER_MINIMUM_VALUE, float(upscaled_conditioning_multiplier)
        )
        upscaled_target_image_width = int(round(primary_target_image_width * conditioning_upscale_factor_clamped))
        upscaled_target_image_height = int(round(primary_target_image_height * conditioning_upscale_factor_clamped))

        zoom_factor_clamped = max(ZOOM_MINIMUM_VALUE, float(zoom))
        offset_x_clamped = _clamp_numeric_value_inclusive(float(offset_x), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE)
        offset_y_clamped = _clamp_numeric_value_inclusive(float(offset_y), OFFSET_MINIMUM_VALUE, OFFSET_MAXIMUM_VALUE)
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

        # Encode once (expensive part); we'll stamp two different SDXL
        # metadata dicts onto separate copies of each raw entry to build
        # the primary and upscaled outputs.
        raw_conditioning_entries_collected_across_all_groups = []
        per_group_full_prompt_text_for_reference_display = []

        if not active_section_descriptors_list:
            # Encode empty prompt and stamp into a single entry per output.
            try:
                empty_conditioning_entry = _encode_one_group_into_one_sdxl_conditioning_entry(
                    clip, [], CLIP_STREAM_PASS_BOTH_L_AND_G, join_separator,
                    mask_token, float(strict_mask), float(start_from_masked),
                    support_a1111_style_embedding_text_setting=bool(support_a1111_style_embedding_text),
                    remove_text_for_unsupported_embeddings_setting=bool(remove_text_for_unsupported_embeddings),
                    filter_known_a1111_embedding_tags_not_installed_locally_setting=bool(filter_known_a1111_embedding_tags_not_installed_locally),
                )
                raw_conditioning_entries_collected_across_all_groups.append(empty_conditioning_entry)
            except Exception as encoding_failure_for_empty_prompt:
                logging.warning(
                    f"CLIPTextEncodeWithCutoffRegionSeparation: empty-prompt encode failed: "
                    f"{encoding_failure_for_empty_prompt}"
                )
                return ([], [], "")
        else:
            grouped_sections_by_clip_pass_choice = _group_active_sections_by_their_clip_pass_choice(
                active_section_descriptors_list
            )

            for clip_pass_choice_for_this_group, section_descriptors_in_this_group in grouped_sections_by_clip_pass_choice.items():
                try:
                    if clip_pass_choice_for_this_group == CLIP_STREAM_PASS_CLASSIC_UPSTREAM_CUTOFF:
                        classic_group_conditioning_list = _encode_one_group_via_classic_upstream_cutoff_plugin(
                            clip,
                            section_descriptors_in_this_group,
                            join_separator,
                            mask_token,
                            float(strict_mask),
                            float(start_from_masked),
                        )
                        for classic_group_entry in classic_group_conditioning_list:
                            raw_conditioning_entries_collected_across_all_groups.append(classic_group_entry)
                    else:
                        group_conditioning_entry = _encode_one_group_into_one_sdxl_conditioning_entry(
                            clip,
                            section_descriptors_in_this_group,
                            clip_pass_choice_for_this_group,
                            join_separator,
                            mask_token,
                            float(strict_mask),
                            float(start_from_masked),
                            support_a1111_style_embedding_text_setting=bool(support_a1111_style_embedding_text),
                            remove_text_for_unsupported_embeddings_setting=bool(remove_text_for_unsupported_embeddings),
                        )
                        raw_conditioning_entries_collected_across_all_groups.append(group_conditioning_entry)
                except Exception as group_encoding_failure:
                    logging.warning(
                        f"CLIPTextEncodeWithCutoffRegionSeparation: group "
                        f"'{clip_pass_choice_for_this_group}' cutoff-style encoding failed "
                        f"({type(group_encoding_failure).__name__}: {group_encoding_failure}). "
                        f"Falling back to plain stock-style encoding for this group (cutoff "
                        f"isolation will not be applied for it)."
                    )
                    try:
                        stock_fallback_conditioning_entry_list = _encode_one_group_via_plain_stock_clip_without_any_isolation(
                            clip,
                            section_descriptors_in_this_group,
                            clip_pass_choice_for_this_group,
                            join_separator,
                        )
                        for stock_fallback_entry in stock_fallback_conditioning_entry_list:
                            raw_conditioning_entries_collected_across_all_groups.append(stock_fallback_entry)
                        full_prompt_text_for_this_group_display = _build_full_prompt_text_for_one_group_of_sections(
                            section_descriptors_in_this_group, join_separator
                        )
                        per_group_full_prompt_text_for_reference_display.append(
                            f"[{clip_pass_choice_for_this_group}] (stock fallback) {full_prompt_text_for_this_group_display}"
                        )
                    except Exception as stock_fallback_failure:
                        logging.warning(
                            f"CLIPTextEncodeWithCutoffRegionSeparation: stock-style fallback "
                            f"for group '{clip_pass_choice_for_this_group}' also failed "
                            f"({type(stock_fallback_failure).__name__}: {stock_fallback_failure}). "
                            f"Skipping this group entirely."
                        )
                    continue

                full_prompt_text_for_this_group_display = _build_full_prompt_text_for_one_group_of_sections(
                    section_descriptors_in_this_group, join_separator
                )
                per_group_full_prompt_text_for_reference_display.append(
                    f"[{clip_pass_choice_for_this_group}] {full_prompt_text_for_this_group_display}"
                )

        # Build the two output CONDITIONING lists by stamping each raw entry
        # with the respective SDXL metadata dict. Tokens tensors are shared
        # by reference (not deep-copied) — only the metadata dict is
        # rebuilt per output.
        primary_output_conditioning_entries = []
        upscaled_output_conditioning_entries = []
        for raw_conditioning_entry in raw_conditioning_entries_collected_across_all_groups:
            raw_entry_tokens_tensor = raw_conditioning_entry[0]
            raw_entry_metadata_dict_unstamped = raw_conditioning_entry[1]

            primary_entry_metadata_dict = dict(raw_entry_metadata_dict_unstamped)
            primary_entry_metadata_dict.update(primary_sdxl_size_and_crop_metadata_fields)
            primary_output_conditioning_entries.append([raw_entry_tokens_tensor, primary_entry_metadata_dict])

            upscaled_entry_metadata_dict = dict(raw_entry_metadata_dict_unstamped)
            upscaled_entry_metadata_dict.update(upscaled_sdxl_size_and_crop_metadata_fields)
            upscaled_output_conditioning_entries.append([raw_entry_tokens_tensor, upscaled_entry_metadata_dict])

        reference_full_prompt_text_for_output = "\n".join(per_group_full_prompt_text_for_reference_display)
        return (
            primary_output_conditioning_entries,
            upscaled_output_conditioning_entries,
            reference_full_prompt_text_for_output,
        )
