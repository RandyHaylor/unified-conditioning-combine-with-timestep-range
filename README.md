# Unified Conditioning Merge (with Timestep Ranges)

A single ComfyUI custom node that fixes a long-standing issue with stock
`ConditioningConcat` and `ConditioningAverage`: they silently drop the
`start_percent` / `end_percent` metadata from the `from`-side input, so a
`ConditioningSetTimestepRange` placed before that input has no effect.

This pack ships a unified merge node plus a small debug node:

- **Unified Conditioning Merge (with timestep ranges)** — replaces stock
  Concat / Combine / Average in one node with a mode dropdown,
  dynamic 1..N inputs, per-slot weights and timestep ranges, and full
  metadata carry-through across both sides.
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

- **split_and_merge_g** (CLIP-G stream): `truncate` / `combine` / `average`
- **split_and_merge_l** (CLIP-L stream): `truncate` / `combine` / `average`

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

### Per-mode behavior (post-splitting)

- `truncate` (default) — keep only the FIRST balanced text piece.
- `combine` — each piece becomes a separate CONDITIONING entry (parallel
  branches the sampler handles independently).
- `average` — encode each piece, then blend the resulting token tensors
  and pooled outputs into a single CONDITIONING entry.

### Output reduction

The two dropdowns are independent. Final output multiplicity:
- If either stream is `combine` → multi-entry CONDITIONING.
- Else if either is `average` → single-entry averaged CONDITIONING.
- Else (both `truncate`) → single-entry truncated CONDITIONING.

Both streams encode in paired pieces; if one stream's split produces fewer
pieces than the other, the shorter side is padded with empty text per stock
`CLIPTextEncodeSDXL`'s shorter-side-pad logic.

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
- `conditioning` (CONDITIONING) — any SDXL-shaped conditioning
- `latent` (LATENT) — used only to read W/H (latent_W * 8, latent_H * 8)
- `zoom` (FLOAT, min 1.0) — how much larger to claim the source frame is
- `offset_x` (FLOAT, -1..+1) — horizontal position of the target window in
  the larger source frame. -1 = far left, 0 = centered, +1 = far right
- `offset_y` (FLOAT, -1..+1) — same vertical

What it writes to each entry's metadata (preserves all other keys like
`start_percent` / `end_percent` / `pooled_output` / `strength`):

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

---

## License

MIT — see `LICENSE`.
