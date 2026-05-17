"""
HTTP routes exposed by this plugin for its frontend extensions.

Routes registered:

  POST /unified-conditioning-merge/validate_prompt_embeddings_sdxl
    Request body JSON:
        {"prompt_texts": ["section1 text", "section2 text", ...]}
    Response body JSON:
        {"messages": [
            "embedding:NAME not found on system",
            "embedding:NAME incompatible with SDXL",
            ...
         ]}
    For each `embedding:NAME` reference found in any of the prompt texts
    (whether the explicit `embedding:NAME` form or an A1111-style bare
    tag that matches a file in the embeddings directory), one classification
    line is emitted:
      - "not found on system" — no file with that stem exists.
      - "incompatible with SDXL" — file exists but its tensor dims don't
        include BOTH 768 (CLIP-L) AND 1280 (CLIP-G).
      - (nothing) — file exists and contains both required dims.

  POST /unified-conditioning-merge/rescan_embeddings_directory
    Forces the embedding-folder scan cache to invalidate and rebuild.
    Response body JSON: {"indexed_file_count": <int>}
    Use after adding/removing embedding files while ComfyUI is running.
"""

import logging
import re

from aiohttp import web
from server import PromptServer

from .embedding_index_scanner import (
    get_cached_or_build_embedding_lowercase_stem_to_index_entry_map,
    invalidate_cached_embedding_filename_to_dim_index,
    is_embedding_file_fully_compatible_with_sdxl_based_on_its_tensor_last_dim_set,
)


EXPLICIT_EMBEDDING_REFERENCE_REGEX_PATTERN = re.compile(r"embedding:([\w./\\-]+)")


def _collect_all_embedding_references_appearing_in_one_prompt_text(
    one_prompt_text, embedding_lowercase_stem_to_index_entry_map
):
    """
    Returns a set of (name_used_in_prompt, lowercase_stem_for_index_lookup,
    bool_is_a1111_bare_style) tuples. Picks up:
      1. Explicit `embedding:NAME` references regardless of whether the file
         actually exists (so we can flag missing ones).
      2. A1111-style bare comma-separated tags (or parenthesized
         `(tag:weight)` forms) but ONLY when they match a known indexed
         filename stem — otherwise plain tags would all light up as
         "embedding not found".
    """
    references_set = set()
    if not one_prompt_text:
        return references_set

    for explicit_match_object in EXPLICIT_EMBEDDING_REFERENCE_REGEX_PATTERN.finditer(one_prompt_text):
        name_string_used_in_prompt = explicit_match_object.group(1)
        references_set.add(
            (name_string_used_in_prompt, name_string_used_in_prompt.lower(), False)
        )

    for raw_comma_separated_part in one_prompt_text.split(","):
        stripped_part_text = raw_comma_separated_part.strip()
        if not stripped_part_text:
            continue
        # Strip optional paren wrapper and weight suffix.
        bare_tag_text_to_consider = stripped_part_text
        if bare_tag_text_to_consider.startswith("(") and bare_tag_text_to_consider.endswith(")"):
            bare_tag_text_to_consider = bare_tag_text_to_consider[1:-1].strip()
            if ":" in bare_tag_text_to_consider:
                bare_tag_text_to_consider = bare_tag_text_to_consider.rsplit(":", 1)[0].strip()
        # Skip if already explicit-form (handled in the regex loop above).
        if bare_tag_text_to_consider.lower().startswith("embedding:"):
            continue
        if bare_tag_text_to_consider.lower() in embedding_lowercase_stem_to_index_entry_map:
            references_set.add(
                (bare_tag_text_to_consider, bare_tag_text_to_consider.lower(), True)
            )

    return references_set


@PromptServer.instance.routes.post(
    "/unified-conditioning-merge/validate_prompt_embeddings_sdxl"
)
async def validate_prompt_embeddings_sdxl_http_route_handler(request):
    try:
        request_body_json_payload = await request.json()
    except Exception:
        return web.json_response(
            {"messages": ["[validation request body was not valid JSON]"]},
            status=400,
        )

    prompt_texts_list = request_body_json_payload.get("prompt_texts", [])
    if not isinstance(prompt_texts_list, list):
        return web.json_response(
            {"messages": ["[validation request must include list-typed 'prompt_texts']"]},
            status=400,
        )

    # Optional fields from the frontend that mirror the runtime node toggles.
    custom_embedding_names_to_strip_text_from_widget = str(
        request_body_json_payload.get("custom_embedding_names_to_strip", "")
    )
    filter_known_a1111_embedding_tags_not_installed_locally_setting_from_widget = bool(
        request_body_json_payload.get("filter_known_a1111_embedding_tags_not_installed_locally", True)
    )

    embedding_index_map = get_cached_or_build_embedding_lowercase_stem_to_index_entry_map()

    # Build the union of "known A1111 embedding names" (file-based curated list
    # + per-node custom widget) to report orphan-tag-strip actions when the
    # corresponding toggle is on at the node.
    union_of_known_a1111_embedding_names_lowercase_set = set()
    if filter_known_a1111_embedding_tags_not_installed_locally_setting_from_widget:
        try:
            from .clip_text_encode_with_cutoff_region_separation import (
                _get_cached_or_lazily_load_known_a1111_embedding_names_lowercase_set,
                _parse_custom_embedding_names_string_into_lowercase_set,
            )
            union_of_known_a1111_embedding_names_lowercase_set = (
                _get_cached_or_lazily_load_known_a1111_embedding_names_lowercase_set()
                | _parse_custom_embedding_names_string_into_lowercase_set(
                    custom_embedding_names_to_strip_text_from_widget
                )
            )
        except Exception:
            pass

    all_embedding_references_combined_across_every_prompt_text = set()
    for one_prompt_text_string in prompt_texts_list:
        if not isinstance(one_prompt_text_string, str):
            continue
        references_in_this_text = _collect_all_embedding_references_appearing_in_one_prompt_text(
            one_prompt_text_string, embedding_index_map
        )
        all_embedding_references_combined_across_every_prompt_text |= references_in_this_text

    output_classification_message_lines = []
    sorted_references_for_stable_output = sorted(
        all_embedding_references_combined_across_every_prompt_text,
        key=lambda triple: (triple[1], triple[0]),
    )
    for (
        name_used_in_prompt,
        lowercase_stem_for_lookup,
        is_a1111_bare_style_reference,
    ) in sorted_references_for_stable_output:
        if lowercase_stem_for_lookup not in embedding_index_map:
            # File doesn't exist locally. If this is a bare tag matching the
            # known-A1111-embedding-names filter list, the runtime will strip
            # it. Otherwise it would fall through to the runtime's
            # `remove_text_for_unsupported_embeddings` path or stock encoder
            # warning. Either way: the embedding will not contribute.
            if (
                is_a1111_bare_style_reference
                and lowercase_stem_for_lookup in union_of_known_a1111_embedding_names_lowercase_set
            ):
                output_classification_message_lines.append(
                    f"Embedding {name_used_in_prompt} not installed locally, "
                    f"will be stripped (orphan A1111 tag filter)"
                )
            else:
                output_classification_message_lines.append(
                    f"Embedding {name_used_in_prompt} not found in system, will be ignored"
                )
            continue
        index_entry_for_this_embedding = embedding_index_map[lowercase_stem_for_lookup]
        if not is_embedding_file_fully_compatible_with_sdxl_based_on_its_tensor_last_dim_set(
            index_entry_for_this_embedding["tensor_last_dim_set"]
        ):
            output_classification_message_lines.append(
                f"Embedding {name_used_in_prompt} incompatible with SDXL, will be ignored"
            )

    return web.json_response({"messages": output_classification_message_lines})


@PromptServer.instance.routes.post(
    "/unified-conditioning-merge/rescan_embeddings_directory"
)
async def rescan_embeddings_directory_http_route_handler(_request):
    invalidate_cached_embedding_filename_to_dim_index()
    rebuilt_index_map = get_cached_or_build_embedding_lowercase_stem_to_index_entry_map()
    return web.json_response({"indexed_file_count": len(rebuilt_index_map)})


logging.info("unified-conditioning-merge: registered HTTP routes for realtime embedding validation.")
