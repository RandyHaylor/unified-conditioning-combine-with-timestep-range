"""
CLIP Text Encode SDXL Enhanced (Detail Isolation) Section

A single-section node intended to be chained together with other instances
and ultimately consumed by a primary `CLIPTextEncodeSDXLEnhancedDetailIsolation`
node. Each section in the chain contributes one chunk to the final prompt.

Chain protocol:
  - Optional input `prompt_sections_in` (`DETAIL_ISOLATION_SECTION_CHAIN`):
    a Python list of section descriptor dicts in chain order. None or
    missing means "this section is the first in the chain".
  - Output `prompt_sections_out` (`DETAIL_ISOLATION_SECTION_CHAIN`):
    the input list with this section's descriptor appended.

Each section descriptor dict carries everything the primary encoder needs
to know about this section, including its own per-section A1111-related
text-transform toggles. Different sections can use different toggle
values (e.g., one section turns off the unsupported-embedding strip while
another keeps it on).

This file is intentionally self-contained — no cross-module imports of
v3's internals. Encoding-time logic lives entirely in the primary node.
"""

DETAIL_ISOLATION_SECTION_CHAIN_TYPE_NAME = "DETAIL_ISOLATION_SECTION_CHAIN"


class CLIPTextEncodeSDXLEnhancedDetailIsolationSection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "global_text": ("STRING", {"multiline": True, "default": ""}),
                "enhanced_text": ("STRING", {"multiline": True, "default": ""}),
                "global_text_weight": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                }),
                "enhanced_text_weight": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                }),
                "clip_l_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "0 = exclude this section from the CLIP-L stream entirely.",
                }),
                "clip_g_strength": ("FLOAT", {
                    "default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05,
                    "tooltip": "0 = exclude this section from the CLIP-G stream entirely.",
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
            },
            "optional": {
                "prompt_sections_in": (DETAIL_ISOLATION_SECTION_CHAIN_TYPE_NAME,),
            },
        }

    RETURN_TYPES = (DETAIL_ISOLATION_SECTION_CHAIN_TYPE_NAME,)
    RETURN_NAMES = ("prompt_sections_out",)
    FUNCTION = "append_this_section_to_incoming_chain_and_return_extended_chain"
    CATEGORY = "unified-conditioning-merge"

    def append_this_section_to_incoming_chain_and_return_extended_chain(
        self,
        global_text,
        enhanced_text,
        global_text_weight,
        enhanced_text_weight,
        clip_l_strength,
        clip_g_strength,
        support_a1111_style_embedding_text,
        remove_text_for_unsupported_embeddings,
        filter_known_a1111_embedding_tags_not_installed_locally,
        prompt_sections_in=None,
    ):
        normalized_global_text_value = " ".join((global_text or "").split()).strip()
        normalized_enhanced_text_value = " ".join((enhanced_text or "").split()).strip()
        new_section_descriptor_for_this_node_instance = {
            "global_text": normalized_global_text_value,
            "enhanced_text": normalized_enhanced_text_value,
            "global_text_weight": float(global_text_weight),
            "enhanced_text_weight": float(enhanced_text_weight),
            "clip_l_strength": float(clip_l_strength),
            "clip_g_strength": float(clip_g_strength),
            "support_a1111_style_embedding_text": bool(support_a1111_style_embedding_text),
            "remove_text_for_unsupported_embeddings": bool(remove_text_for_unsupported_embeddings),
            "filter_known_a1111_embedding_tags_not_installed_locally": bool(
                filter_known_a1111_embedding_tags_not_installed_locally
            ),
        }
        existing_chain_or_empty_list = (
            list(prompt_sections_in) if prompt_sections_in is not None else []
        )
        existing_chain_or_empty_list.append(new_section_descriptor_for_this_node_instance)
        return (existing_chain_or_empty_list,)
