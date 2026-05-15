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

- **merge_mode** (dropdown) — one of `concat` / `combine` / `average` /
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

---

## The four modes (and what they mean for the sampler)

| Mode                 | What it does                                                                                                       |
|----------------------|--------------------------------------------------------------------------------------------------------------------|
| `concat`             | **Glues prompts end-to-end into one longer prompt per timestep segment.** Tokens are concatenated; the sampler reads ONE longer conditioning.       |
| `combine`            | **Hands the prompts to the sampler as separate parallel branches.** The sampler runs the model once per active branch each step and blends results. |
| `average`            | **Mathematically blends the embeddings into one prompt per timestep segment** using your weights. NOT normalized — can boost or suppress overall.    |
| `average_normalized` | Same as `average` but divides by the sum of active weights so the output magnitude stays comparable to a single prompt.                              |

### Two fundamentally different shapes

This matters when chaining. Modes produce two different output shapes:

- **Concat / average / average_normalized** produce one output **segment**
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
| `average`            | Each input's contribution is scaled by `weight_i` in the weighted sum, with NO division by total weight.        |
| `average_normalized` | Each input's contribution is `weight_i / sum(active_weights)` — proper weighted average.                        |

Default `weight = 1.0` keeps everything in stock-compatible territory.

---

## Timestep range segmentation (concat / average modes)

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

1. Change THIS node's mode from `concat` / `average` / `average_normalized`
   to `combine`. Parallel branches pass through.
2. Change the UPSTREAM node from `combine` to `concat` / `average` /
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
upstream concat / average) on a slot — non-overlapping — are fine and
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

## License

MIT — see `LICENSE`.
