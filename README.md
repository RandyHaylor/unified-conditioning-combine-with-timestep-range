# Unified Conditioning Merge (with Timestep Ranges)

A single ComfyUI custom node that fixes a long-standing issue with stock
`ConditioningConcat` and `ConditioningAverage`: they silently drop the
`start_percent` / `end_percent` metadata from the `from`-side input, so a
`ConditioningSetTimestepRange` placed before that input has no effect.

This pack ships:

- **Unified Conditioning Merge (with timestep ranges)** — replaces stock
  Concat / Combine / Average in one node with a mode dropdown,
  dynamic 1..N inputs, per-slot weights and timestep ranges, and full
  metadata carry-through across both sides.
- **CLIPTextEncodeSDXL (auto-split-and-merge)** — drop-in extension of
  stock SDXL text encoder with smart splitting at `BREAK` / comma / line /
  whitespace boundaries when a prompt exceeds 77 tokens. Per-stream
  concat / truncate / combine / average dropdowns (default concat — matches
  stock SDXL encoder's long-prompt handling).
- **Conditioning-crop-zoom-SDXL** — rewrites SDXL size/crop conditioning
  metadata using the latent's actual W/H plus a zoom factor and offset.
  Recommended as the FINAL stop on both positive and negative
  conditioning before the sampler (see its section below).
- **CLIP Text Encode (Cutoff Region Separation)** — single-node prompt builder
  with a `section_count` you can expand/collapse manually. Each section
  has a text field and an `isolate` toggle that uses the Cutoff algorithm
  to confine that section's influence to its own tokens (prevents prompt
  context bleeding). Requires the ComfyUI_Cutoff plugin.
- **Debug Conditioning** — in-line passthrough that prints CONDITIONING
  structure to the server console and emits a STRING dump output.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RandyHaylor/unified-conditioning-combine-with-timestep-range.git unified-conditioning-merge
```

Restart ComfyUI. Both nodes appear under the **unified-conditioning-merge**
category.

---

## Merge node UI

- **merge_mode** (dropdown) — one of `concat` / `combine` / `average_additive` /
  `average_normalized`. See below.
- **conditioning_N** (input slot) — a CONDITIONING input. The node always
  has exactly one trailing empty slot; connect it and a new empty slot
  appears. Disconnect any slot and it auto-collapses.
- **conditioning_N_start**, **_end** (FLOAT 0..1) — per-input timestep window.
  Widget ranges **intersect** with any upstream `start_percent` /
  `end_percent` already on the input, so:
  - Widget `[0, 1]` is a true pass-through; upstream ranges fully control.
  - The widget can only **narrow** the upstream range, never widen it.
- **conditioning_N_weight** (FLOAT 0..10, default 1.0) — per-input weight.
  Meaning depends on mode (see below). Default `1.0` reproduces stock
  behavior in every mode.
- **conditioning_N_clip** (dropdown: `Pass L+G` / `Pass L` /
  `Pass G`, default `Pass L+G`) — SDXL-only. Lets you route this input's
  prompt to only one of SDXL's two CLIP streams by zeroing the other half
  of each token's embedding. `Pass L` also zeros `pooled_output` since
  pooled is CLIP-G only. Non-SDXL conditioning (last embedding dim != 2048)
  ignores this setting and passes through unchanged.

---

## The four modes (and what they mean for the sampler)

| Mode                 | What it does                                                                                                       |
|----------------------|--------------------------------------------------------------------------------------------------------------------|
| `concat`             | **Glues prompts end-to-end into one longer prompt per timestep segment.** Tokens are concatenated; the sampler reads ONE longer conditioning.       |
| `combine`            | **Hands the prompts to the sampler as separate parallel branches.** The sampler runs the model once per active branch each step and blends results. |
| `average_additive`   | **Mathematically blends the embeddings into one prompt per timestep segment** using your weights as raw additive coefficients. NOT normalized — can boost or suppress overall magnitude. |
| `average_normalized` | Same blend as `average_additive` but divides by the sum of active weights so the output magnitude stays comparable to a single prompt.                                                  |

### Two fundamentally different shapes

This matters when chaining. Modes produce two different output shapes:

- **Concat / average_additive / average_normalized** produce one output **segment**
  per non-overlapping timestep interval, with each segment containing ONE
  cond. Sequential in time. The sampler picks one segment per step.
- **Combine** produces one output **branch** per input, all carrying their
  own timestep windows that may overlap. Parallel in time. The sampler may
  process multiple branches per step.

---

## Per-slot weight semantics (per mode)

| Mode                 | Weight effect                                                                                                   |
|----------------------|-----------------------------------------------------------------------------------------------------------------|
| `concat`             | Tokens are pre-scaled by `weight_i` before concatenation (`torch.cat([w_i * t_i for active], dim=1)`). Equivalent to CLIP `(token:weight)` syntax. |
| `combine`            | Each emitted entry gets `metadata['strength'] = weight_i` (only if != 1.0). Equivalent to chaining `ConditioningSetAreaStrength` on each input.   |
| `average_additive`   | Each input's contribution is scaled by `weight_i` in the weighted sum, with NO division by total weight.        |
| `average_normalized` | Each input's contribution is `weight_i / sum(active_weights)` — proper weighted average.                        |

Default `weight = 1.0` keeps everything in stock-compatible territory.

---

## Timestep range segmentation (concat / average_additive / average_normalized modes)

When inputs have different effective ranges, the node breaks the [0, 1]
timestep span at every endpoint and emits one CONDITIONING entry per
sub-interval. Only inputs whose effective range covers a sub-interval
contribute to it.

**Example:** two inputs, widget ranges `[0, 1]`, upstream
`conditioning_1 = [0.0, 0.6]`, `conditioning_2 = [0.3, 1.0]`:

| Range        | Active subset | concat output                                            | average_normalized output (w=1, w=1) |
|--------------|---------------|----------------------------------------------------------|---------------------------------------|
| [0.0, 0.3]   | {1}           | `conditioning_1` tokens                                  | `conditioning_1` tokens              |
| [0.3, 0.6]   | {1, 2}        | `cat(c1_tokens, c2_tokens)`                              | `0.5 * c1 + 0.5 * c2`                |
| [0.6, 1.0]   | {2}           | `conditioning_2` tokens                                  | `conditioning_2` tokens              |

Combine mode does NOT segment — it emits each input narrowed to its
effective range as a parallel branch.

---

## Chaining rules — important!

This is where intent and shape have to match.

### Combine → combine → ... → sampler  ✓
Parallel branches propagate cleanly. Each combine may add more branches and
each input's ranges narrow further. Sampler handles the final list.

### Concat → concat (with non-overlapping ranges) ✓
The first concat produces sequential time segments. Feeding into a second
concat with another input adds more breakpoints and re-glues. Works fine —
each input slot has at most one active entry per sub-interval.

### Concat → combine  ✓
Concat's sequential output becomes one of several parallel branches. Fine.

### **Combine → concat  ✗ NOW BLOCKED**

This was the silently-destructive case. Combine produces overlapping parallel
branches; concat glues tokens into one prompt per segment, which collapses
those parallel branches into a single merged-tokens cond. Your upstream
combine has no effect on the image because its parallel intent is erased.

**As of this release, the node refuses to run in this configuration and
raises a clear error pointing at the offending slot.** Wire-up is still
allowed (so the graph can stay flexible while you tinker), but the actual
prompt execution fails until you fix the structure.

The error names the slot, prints the two overlapping ranges, and lists three
ways to fix it:

1. Change THIS node's mode from `concat` / `average_additive` /
   `average_normalized` to `combine`. Parallel branches pass through.
2. Change the UPSTREAM node from `combine` to `concat` / `average_additive` /
   `average_normalized`. The earlier split is replaced with token-level
   merging, and this node sees only one entry per slot.
3. Restructure so `combine` mode is kept all the way out to the sampler;
   never put a flat-merge node downstream of a combine.

### Why we refuse rather than warn
Flat-flatten-after-combine is creatively destructive and silent. Changes you
make on an upstream combine produce no visible change in the image — a
debugging nightmare. We'd rather you find out at run time with a clear error
than wonder why your knob isn't doing anything.

### Detection rule
"Overlapping ranges within a single slot" is the signature of parallel
branches. The flat modes refuse if ANY single slot has two or more entries
whose effective timestep windows overlap. Sequential entries (segments from
upstream concat / average_additive / average_normalized) on a slot — non-overlapping — are fine and
common, so concat → concat chains stay supported.

---

## Compatibility

`comfy/samplers.py:calculate_start_end_timesteps` reads each entry's
`start_percent` / `end_percent` and computes the correct sigma window — so
every standard sampler (KSampler, KSamplerAdvanced, WAS_KSampler, etc.)
honors per-entry ranges natively. No special integration needed; just drop
the node in.

---

## Debug node

`Debug Conditioning` is an in-line passthrough that:
- Prints the full CONDITIONING structure (shapes, dtypes, devices, min/max,
  every metadata key) to the server console under a label you supply.
- Returns the same CONDITIONING unchanged, plus a STRING output (`debug_text`)
  with the same dump for wiring into a ShowText / SaveText node.
- Marked `OUTPUT_NODE = True` so you can run workflows that end in it.

Useful for verifying that `start_percent` / `end_percent` / `pooled_output` /
`strength` survive each step of a chain.

---

## CLIPTextEncodeSDXL (auto-split-and-merge)

A drop-in extension of stock `CLIPTextEncodeSDXL` with smart pre-tokenize
splitting AND two dropdowns controlling what to do with the resulting
chunks per stream:

- **split_and_merge_g** (CLIP-G stream): `concat` / `truncate` / `combine` / `average`
- **split_and_merge_l** (CLIP-L stream): `concat` / `truncate` / `combine` / `average`

Default for both is `concat` — reproduces stock SDXL encoder's "encode every
chunk in one call, concat outputs along the sequence dim" behavior. Pick a
non-default mode only when you want a specific deviation from stock.

The pipeline that runs is determined by the highest-priority mode across
both streams, in this order: **combine > average > concat > truncate**. So
if either side picks `combine`, the combine pipeline runs and produces a
multi-entry output. If either picks `average` (and neither picks combine),
the result is a single blended entry. If both pick `concat`, the stock-style
single-encode pipeline runs. If both pick `truncate`, only the first chunk
pair encodes (overflow dropped).

### How the splitting works

The node does its own text splitting BEFORE tokenization to avoid the
"tiny last chunk" problem (a 4-token tail gets way over-weighted in average
mode and looks goofy in combine mode). Split marker priority, highest first:

1. **`BREAK`** — uppercase whole word, same convention as A1111 / Forge.
   Hard boundary — sub-balancing never crosses it.
2. **comma** `,`
3. **line break** `\n`
4. **any whitespace**
5. **character force-split** (last resort)

For each `BREAK`-separated segment, the node counts its content tokens and
computes `K = ceil(tokens / 75)`, the number of 75-token sub-chunks that
segment needs. If `K == 1` the segment is used as-is. If `K > 1` it sub-
splits at the highest-priority delimiter where every resulting piece fits,
then greedy-bin-packs the pieces into K bins targeting equal token counts.
Result: chunks are roughly the same size — no tiny over-weighted tail.

### Per-mode behavior

- `concat` (default) — feeds all tokenized chunks (respecting BREAK as a
  forced chunk boundary) to one encode call, returning one CONDITIONING
  entry whose token sequence is `77 * N` long. Matches stock SDXL encoder.
  **No balanced sub-splitting** — uses the tokenizer's natural greedy
  chunking so chunk-padding patterns match what the model was trained on.
- `truncate` — drop overflow; keep only the FIRST 77-token chunk for that
  stream.
- `combine` — sub-split text into balanced pieces (BREAK-respecting +
  balanced via comma/line/space cascade); each piece becomes a separate
  CONDITIONING entry (parallel branches the sampler handles independently).
- `average` — sub-split text into balanced pieces; encode each piece, then
  blend the resulting token tensors and pooled outputs into a single
  CONDITIONING entry.

### Pipeline priority (when g and l pick different modes)

The two dropdowns are independent; if they disagree, the **higher-priority**
mode wins and dictates the pipeline that runs:

**combine > average > concat > truncate**

So with `g=combine, l=concat`, the combine pipeline runs (per-piece encode,
multi-entry output). With `g=concat, l=average`, the average pipeline runs.
With both `concat`, the stock-style pipeline runs. With both `truncate`,
only the first chunk pair encodes.

### Padding for stream length mismatches

Within any pipeline, if one stream produces fewer chunks/pieces than the
other, the shorter side is padded with empty chunks (matches stock
`CLIPTextEncodeSDXL`'s pad-shorter-side logic).

### Inputs

Same size-conditioning fields as stock SDXL encoder (`width`, `height`,
`crop_w/h`, `target_width/height`), plus `text_g`, `text_l`, and the two
dropdowns above.

### Outputs

- `conditioning` (CONDITIONING)
- `debug_info` (STRING) — names balanced-piece counts per stream, the modes,
  per-pair token counts, and the chosen reduction strategy. Wire into a
  ShowText node to see exactly how splitting happened.

---

## Conditioning-crop-zoom-SDXL

Modifies SDXL size/crop conditioning metadata on every CONDITIONING entry
to create a **framing/composition hint** ("free zoom bias"). It does NOT
crop or zoom the latent itself — it just rewrites the metadata that SDXL's
conditioning pipeline reads.

Inputs:
- `positive` (CONDITIONING, required) — positive-prompt conditioning
- `negative` (CONDITIONING, optional) — negative-prompt conditioning; gets
  the same metadata written. If unconnected, the `negative` output is empty.
- `latent` (LATENT) — used only to read W/H (latent_W * 8, latent_H * 8)
- `zoom` (FLOAT, min 1.0) — how much larger to claim the source frame is
- `offset_x` (FLOAT, -1..+1) — horizontal position of the target window in
  the larger source frame. -1 = far left, 0 = centered, +1 = far right
- `offset_y` (FLOAT, -1..+1) — same vertical

Outputs:
- `positive` (CONDITIONING) — same as input with size/crop metadata updated.
- `negative` (CONDITIONING) — same.

What it writes to each entry's metadata on BOTH pos and neg (preserves
all other keys like `start_percent` / `end_percent` / `pooled_output` /
`strength`):

| Flat key (ComfyUI form)         | Tuple key (SDXL paper form)     |
|---------------------------------|---------------------------------|
| `target_width` / `target_height` | `target_size_as_tuple`          |
| `width` / `height`              | `original_size_as_tuple`        |
| `crop_w` / `crop_h`             | `crop_coords_top_left`          |

Math:
- `target = latent W/H`
- `source = round(latent W/H * zoom)`
- `crop = (offset+1)/2 * (source - target)` per axis

Example: latent=1024², zoom=2.0 → source=2048², max crop window=1024².
- `offset_x=0, offset_y=0` → crop=(512, 512) (centered)
- `offset_x=-1, offset_y=+1` → crop=(0, 1024) (top-left of source, but
  framed to show the bottom-right of the implied larger image)

### Recommended placement: ALWAYS the final stop before the sampler

Put this node at the very end of your conditioning graph, with **both
positive and negative** wired through it, and connect its two outputs to
the sampler's `positive` and `negative` inputs respectively. Two reasons:

1. **Convenience / safety:** it auto-derives `target_width` / `target_height`
   from the latent you're sampling against, so you can't accidentally
   drift between rendered size and conditioning size when changing
   resolution. Stock `CLIPTextEncodeSDXL` hardcodes these to defaults
   that may not match your latent.
2. **Quality bias:** even at default `zoom=1.0, offset=0` you've fixed
   the conditioning's W/H to the actual latent W/H. With `zoom > 1.0`
   you get a "free zoom bias" — the model is told the rendered frame is
   a sub-crop of a larger source image, which empirically often produces
   better-composed and higher-detail output (SDXL was trained with
   non-zero crop coords on a large fraction of its data, so claiming
   `crop=0` with `source=target` puts you in a thinly-trained slice of
   its conditioning distribution).

Setting it to default (`zoom=1.0, offsets=0`) is already an improvement
over not using it at all, because it pins target/source to your real
latent size on BOTH pos and neg in one node.

---

## CLIP Text Encode (Cutoff Region Separation)

A single-node prompt builder built on top of BlenderNeko's
[ComfyUI_Cutoff](https://github.com/BlenderNeko/ComfyUI_Cutoff). The Cutoff
algorithm prevents prompt context bleeding by measuring each phrase's
influence on the global CLIP embedding, then confining that influence back
to the phrase's own tokens. Read the linked README for the underlying
theory.

This node also bundles:
- **Per-section CLIP stream routing** (Pass L+G / Pass L / Pass G) — sections
  are grouped by their stream choice, each group runs through Cutoff
  independently, the resulting tensors are stream-masked (zero the L or G
  half), and the per-group outputs are emitted as a multi-entry CONDITIONING
  (combine-style — sampler treats each group as a parallel branch).
- **SDXL size/crop metadata** — optional LATENT input determines target W/H
  (defaults to 1024x1024 when LATENT is unconnected). `zoom` + `offset_x`
  + `offset_y` widgets compute `width`/`height`/`crop_w`/`crop_h` per the
  Conditioning-crop-zoom-SDXL math, applied to every output entry.

**Self-contained — no external plugin dependency.** The Cutoff algorithm
itself (mask targets, encode masked variant, blend per-token by region) is
re-implemented from scratch in `cutoff_per_stream_isolation.py` under MIT
license. This means we can run the algorithm **independently per CLIP
stream** (L and G), which lets per-section CLIP routing actually work —
when a section is "L only", the L stream encodes that section's text and
the G stream encodes the empty prompt (natural CLIP-G empty embedding,
not zero-masking).

### UI

- `clip` (CLIP) — the CLIP model to encode through.
- `section_count` (INT 1..16, default 3) — how many section pairs are
  active. Section widgets with index > section_count are hidden in the UI
  (their values still persist in the workflow JSON, so you can re-expand
  without losing typed text).
- `join_separator` (STRING, default `","`) — how the active sections are
  joined into the full prompt text that CLIP sees. Comma is the default
  CLIP tag separator; switch to `\n` if you want per-line regions
  (which is also a natural Cutoff boundary).
- `mask_token` (STRING, default `""`) — passed through to Cutoff's finalize.
- `strict_mask` (FLOAT 0..1, default `1.0`) — Cutoff strictness.
  `1.0` = each isolated phrase only affects its own region;
  `0.0` = phrases can still affect outside specified areas.
- `start_from_masked` (FLOAT 0..1, default `1.0`) — passed through to
  Cutoff's finalize.
- `section_N_text` (STRING multiline) — this section's prompt text.
- `section_N_isolate` (BOOLEAN, default `True`) — if true, this section's
  tokens are registered as a Cutoff region with
  `region_text == target_text == section text` (phrase-level
  decontamination). If false, the section is still included in the
  full prompt but not isolated.
- `section_N_weight` (FLOAT -10..10, default `1.0`) — applied two
  different ways depending on `section_N_isolate`:
  - **isolate=True** → passed to Cutoff's `add_clip_region` as the region
    weight (range and default match stock `CLIPSetRegion`). The section's
    text appears unwrapped in the full prompt.
  - **isolate=False** → wraps the section's text in CLIP's attention
    syntax `(text:weight)` in the full prompt (unless weight is 1.0, in
    which case the wrap is skipped). The section is still part of the
    prompt CLIP encodes; no Cutoff region is registered.

  Default `weight = 1.0` produces the same effect either way: bare text,
  no region weight, no attention wrap.

- `section_N_clip` (dropdown: `Pass L+G` / `Pass L` / `Pass G` / `Classic`,
  default `Pass L+G`) — which SDXL CLIP stream(s) this section's tokens
  contribute to. Sections sharing a choice get grouped into one encoding
  pass. You can mix modes within one node — e.g., two sections Classic
  + one section Pass L — they are encoded by their respective code paths
  and combined into a multi-entry parallel-branch CONDITIONING.
  - **`Pass L+G`** (default): section's text goes to both CLIP-L and CLIP-G
    via our self-contained per-stream Cutoff implementation. No external
    plugin dependency. Verified in testing to produce equivalent quality
    to (and in some cases observably better than) the upstream Cutoff
    plugin's output for the common single-prompt case.
  - **`Pass L`**: section's text goes to CLIP-L only; CLIP-G for the
    section's group is encoded as the empty prompt (giving CLIP-G's
    natural empty-prompt embedding — NOT a zero-masked tensor, which
    would be out-of-distribution for SDXL). Use to route tag-style or
    short prompts to L while leaving G's natural baseline intact.
  - **`Pass G`**: symmetric — section's text goes to CLIP-G only;
    CLIP-L encodes the empty prompt. Useful for long natural-language
    prompts that the larger CLIP-G encoder handles better.
  - **`Classic`**: routes through the upstream
    [ComfyUI_Cutoff](https://github.com/BlenderNeko/ComfyUI_Cutoff) plugin
    directly. Verified to reproduce the upstream plugin's exact behavior
    when used standalone (since this mode IS the upstream plugin's code).
    Use when you want the canonical Cutoff behavior, or for side-by-side
    comparison against the Pass L+G / L / G modes. **Requires
    ComfyUI_Cutoff to be installed** — the other three modes have no
    external dependency.

- `zoom` (FLOAT min 1.0, default 1.0) — SDXL "source frame larger than
  rendered frame" zoom bias. Same semantics as the standalone
  `Conditioning-crop-zoom-SDXL` node.
- `offset_x` / `offset_y` (FLOAT -1..1, default 0) — position the
  rendered frame inside the implied larger source. -1 = far left/top,
  +1 = far right/bottom.
- `latent` (LATENT, optional) — read for image-space W/H to use as the
  SDXL target size for the primary `conditioning` output. If unconnected,
  defaults to 1024×1024.
- `conditioning_upscale_by` (FLOAT, min 1.0, default 1.0) — multiplier
  applied to the primary target W/H to derive the target W/H for the
  `upscaled_conditioning` output. Computed in Python — does NOT take a
  downstream upscaled latent as input (ComfyUI validates the entire graph
  upfront, before any node runs, so a downstream latent doesn't exist at
  validation time). At `1.0` (default), `upscaled_conditioning` is
  identical to the primary `conditioning`. At e.g. `1.5`, upscaled
  metadata uses W*1.5 / H*1.5 — matching a typical hires-fix latent
  upscale by 1.5x.

Outputs:
- `conditioning` (CONDITIONING) — primary, sized to `latent`'s W/H.
- `upscaled_conditioning` (CONDITIONING) — same prompt encoding, sized to
  `upscaled_latent`'s W/H (or identical to primary if no/same upscale).
- `reference_full_prompt` (STRING) — debug dump of per-group prompts.

This lets a single prompt node drive a **two-stage workflow** (base
sampler at one resolution, then a hires-fix / upscaler sampler at a
larger resolution) without having to maintain two copies of the same
prompt with slightly different target sizes.

Outputs:
- `conditioning` (CONDITIONING) — finalized through Cutoff; wire to a sampler.
- `reference_full_prompt` (STRING) — the joined prompt text that CLIP
  actually encoded; wire into ShowText to verify what you built.

### Skipped sections

Empty `section_N_text` (only whitespace, or blank) is skipped even within
the active range — so you can leave some slots blank without contaminating
the prompt.

### Workflow shape

```
[Section 1: woman portrait]
[Section 2: warm lighting]   --> Cutoff Sections Prompt --> KSampler positive
[Section 3: city background]                              \
                                                           +--> ShowText (reference_full_prompt)
```

All sections live on the same node face; increase / decrease
`section_count` to expand or collapse.

---

## License

MIT — see `LICENSE`.
