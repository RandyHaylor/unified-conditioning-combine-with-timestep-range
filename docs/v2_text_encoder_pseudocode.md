# V2 Text Encoder — Pseudocode and Design

A from-scratch redesign of the cutoff-region-separation text encoder for
SDXL. Replaces the v1 cutoff-paper algorithm with a simpler split-by-
isolation-amount model.

## Design rules

- **Single region type.** Every region has the same widget set. No
  `isolate` toggle. The `isolation_amount` float (0..1) controls how
  private vs. global the region is.
- **Categorization rule for endpoints:**
  - `isolation_amount = 0.0` → region appears in the GLOBAL stack only.
    NOT in the isolated list (not even at weight 0).
  - `isolation_amount = 1.0` → region appears in the ISOLATED list only.
    NOT in the global stack.
  - `0.0 < isolation_amount < 1.0` → region appears in BOTH lists with
    fractional weights `(1 - isolation_amount) * weight` and
    `isolation_amount * weight` respectively.
- **L and G handled independently** end-to-end. Each stream has its own
  text construction, encoding pass, and per-region embedding overlay.
- **Per-stream zero-strength = exclude entirely.** `clip_l_strength = 0`
  means this region's text never enters the CLIP-L stream's prompt at
  all (not multiply-by-zero). Avoids the `(tag:0)` paradox.
- **Single composite CONDITIONING output**, position-mapped like v1.
  Output shape matches stock SDXL encoder.
- **SDXL-only.** Drops "classic upstream cutoff" path. Drops per-section
  `clip_pass` dropdown.

## Per-region widget set

- `region_N_text` — STRING multiline.
- `region_N_weight` — FLOAT, default 1.0, range -10..10.
- `region_N_isolation_amount` — FLOAT, default 1.0, range 0..1.
- `region_N_clip_l_strength` — FLOAT, default 1.0, range -10..10. 0 = exclude from L.
- `region_N_clip_g_strength` — FLOAT, default 1.0, range -10..10. 0 = exclude from G.
- `region_N_weight_from_other_isolated_regions` — FLOAT, default 0.0, range -10..10.

## Encoding pseudocode

```
def encode_v2(regions, sdxl_size_and_crop_metadata):
    # 1. Categorize regions per the endpoint exclusion rule
    regions_appearing_in_global_stack = [
        R for R in regions if R.isolation < 1.0
    ]
    regions_appearing_in_isolated_list = [
        R for R in regions if R.isolation > 0.0
    ]

    # 2. Independent processing per CLIP stream (L, G)
    final_per_stream_token_embedding_tensors = {}
    final_per_stream_pooled_outputs = {}

    for stream_key in ["l", "g"]:

        # 2a. Build the global stack text for this stream.
        # Each global contributor wrapped at
        #   R.weight * (1 - R.isolation) * R.stream_strength.
        # Regions with stream_strength == 0 are excluded entirely.
        global_stack_text_parts = []
        for region_global in regions_appearing_in_global_stack:
            per_stream_strength = stream_strength_of(region_global, stream_key)
            if per_stream_strength == 0:
                continue
            global_contribution_weight = (
                region_global.weight
                * (1.0 - region_global.isolation)
                * per_stream_strength
            )
            global_stack_text_parts.append(
                f"({region_global.text}:{global_contribution_weight})"
            )

        # 2b. For each region, build the per-region encoding prompt:
        #   global stack
        # + own isolated portion (if isolation > 0)
        # + each OTHER isolated region pulled in at scaled weight
        per_region_encoded_embedding_for_this_stream = {}
        for region in regions:
            per_stream_strength = stream_strength_of(region, stream_key)
            if per_stream_strength == 0:
                per_region_encoded_embedding_for_this_stream[region.id] = None
                continue

            per_region_prompt_parts = list(global_stack_text_parts)

            if region.isolation > 0:
                own_isolated_weight = (
                    region.weight
                    * region.isolation
                    * per_stream_strength
                )
                per_region_prompt_parts.append(
                    f"({region.text}:{own_isolated_weight})"
                )

            if region.weight_from_other_isolated_regions != 0:
                for other_isolated in regions_appearing_in_isolated_list:
                    if other_isolated.id == region.id:
                        continue
                    other_stream_strength = stream_strength_of(other_isolated, stream_key)
                    if other_stream_strength == 0:
                        continue
                    pulled_in_weight = (
                        region.weight_from_other_isolated_regions
                        * other_isolated.weight
                        * other_isolated.isolation
                        * other_stream_strength
                    )
                    per_region_prompt_parts.append(
                        f"({other_isolated.text}:{pulled_in_weight})"
                    )

            per_region_full_prompt = ", ".join(per_region_prompt_parts)
            per_region_tokens = clip.tokenize(per_region_full_prompt)[stream_key]
            encoded_embedding = stream_specific_encoder_for(stream_key).encode(
                per_region_tokens
            )
            per_region_encoded_embedding_for_this_stream[region.id] = encoded_embedding

        # 2c. Canonical output sequence: raw text of all regions in order,
        # comma-joined, NO weights. Defines output token positions.
        canonical_raw_joined_text = ", ".join(R.text for R in regions)
        canonical_tokens = clip.tokenize(canonical_raw_joined_text)[stream_key]
        canonical_encoding = stream_specific_encoder_for(stream_key).encode(canonical_tokens)

        # 2d. Identify each region's token-position range within canonical
        # by sublist matching.
        per_region_canonical_position_range = {}
        canonical_content_ids = flatten_content_token_ids(canonical_tokens)
        for region in regions:
            per_stream_strength = stream_strength_of(region, stream_key)
            if per_stream_strength == 0:
                per_region_canonical_position_range[region.id] = None
                continue
            region_own_tokens = clip.tokenize(region.text)[stream_key]
            region_own_content_ids = flatten_content_token_ids(region_own_tokens)
            position_range = find_first_sublist_match(
                canonical_content_ids, region_own_content_ids
            )
            per_region_canonical_position_range[region.id] = position_range

        # 2e. Compose final embedding: start from canonical encoding for
        # structural tokens; overlay each region's own encoding at its
        # canonical position range.
        final_embedding = canonical_encoding.clone()

        for region in regions:
            region_encoding = per_region_encoded_embedding_for_this_stream[region.id]
            canonical_position_range = per_region_canonical_position_range[region.id]
            if region_encoding is None or canonical_position_range is None:
                continue

            # Locate region's text within its own per-region encoding
            region_text_pos_in_own_encoding = find_first_sublist_match(
                flatten_content_token_ids_of(region_encoding_tokens_for(region)),
                region_own_content_ids,
            )
            if region_text_pos_in_own_encoding is None:
                continue

            cs, ce = canonical_position_range
            rs, re = region_text_pos_in_own_encoding
            final_embedding[:, cs:ce, :] = region_encoding[:, rs:re, :]

        final_per_stream_token_embedding_tensors[stream_key] = final_embedding

        if stream_key == "g":
            final_per_stream_pooled_outputs["g"] = canonical_pooled_for(canonical_encoding)

    # 3. SDXL combine: concat L (768) + G (1280) along last dim.
    sdxl_combined = torch.cat(
        [
            final_per_stream_token_embedding_tensors["l"],
            final_per_stream_token_embedding_tensors["g"],
        ],
        dim=-1,
    )

    # 4. Single CONDITIONING entry with SDXL metadata.
    output_metadata = dict(sdxl_size_and_crop_metadata)
    output_metadata["pooled_output"] = final_per_stream_pooled_outputs["g"]
    return [[sdxl_combined, output_metadata]]
```

## Open issues / decisions deferred to implementation

1. **Sublist matcher robustness.** BPE-context-sensitive tokenization
   can prevent a region's tokens from matching at the canonical level.
   Reuse v1's `_find_all_sublist_match_start_positions_within_superlist`
   plus the tensor-safe equality helpers; same edge cases apply.
2. **Structural tokens (commas, BOS, EOS, padding) at non-region positions**
   inherit from the canonical encoding. They "see" every region's text
   via attention during canonical encoding. No `strict_mask`-equivalent
   slider in the spec; if needed later, add a "non-region isolation"
   knob that swaps canonical encoding for a globals-only encoding at
   non-region positions.
3. **Pooled output** taken from canonical encoding's G stream. Simplest
   default; matches stock SDXL practice. Could be replaced with per-
   region pooled blend later.
4. **Region count widget.** Default 2, range 0-32. Implementation can
   start with statically-declared 32 slots + a region_count widget that
   determines how many are "active"; later upgrade to dynamic-slot UI
   like the unified merge node.
5. **Embedding shape rescue + name validation.** Same as v1: lift the
   helpers from `cutoff_per_stream_isolation.py` and `server_routes.py`
   to keep prompt-validation parity.

## Naming / file plan

- Python: `clip_text_encode_sdxl_v2_with_isolation_amount.py`
- Display name: `CLIP Text Encode SDXL v2 (isolation amount)`
- Class name: `CLIPTextEncodeSDXLV2WithIsolationAmount`
- Frontend JS (later): `web/clip_text_encode_sdxl_v2_dynamic_region_slots.js`
- Category: `unified-conditioning-merge`
- V1 node stays untouched at `clip_text_encode_with_cutoff_region_separation.py`.
