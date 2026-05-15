# Conditioning Merge (with Timestep Ranges)

A single ComfyUI custom node that fixes a long-standing issue with stock
`ConditioningConcat` and `ConditioningAverage`: they silently drop the
`start_percent` / `end_percent` metadata from the `from`-side input, so a
`ConditioningSetTimestepRange` placed before that input has no effect.

This node replaces all of stock Concat / Combine / Average with one
dynamic-input node that:

- Carries `start_percent`/`end_percent` forward across every input.
- Exposes per-input range and weight widgets.
- Accepts any number of inputs (1–N), auto-adding a trailing empty slot.
- Picks merge behavior with a four-option dropdown.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RandyHaylor/unified-conditioning-combine-with-timestep-range.git conditioning_merge_nodes_with_timestep_range
```

Restart ComfyUI. The node appears under **advanced/conditioning** as
**"Conditioning Merge (with timestep ranges)"**.

## Node UI

- **merge_mode** (dropdown): `concat` / `combine` / `average` / `average_normalized`.
- **conditioning_N** (input slot): a CONDITIONING input. The node always has
  exactly one trailing empty slot — connect it and a new empty slot appears.
- **conditioning_N_start**, **_end** (FLOAT, 0.0–1.0): per-input timestep
  window. Widget ranges **intersect** with any upstream
  `start_percent`/`end_percent` already on that input's metadata dict, so:
  - Widget `[0, 1]` is a true pass-through — upstream ranges fully control output.
  - Widget can only **narrow** the upstream range, never widen it.
- **conditioning_N_weight** (FLOAT, 0.0–10.0, default 1.0): per-input weight.
  Meaning depends on `merge_mode` (see below). Default `1.0` reproduces
  stock-node behavior in every mode.

## Modes

| Mode                 | Behavior for each segment's active subset                                                                                                          |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `concat`             | `torch.cat([weight_i * tokens_i for i in active], dim=1)` in slot order. Pre-scaling tokens by weight is equivalent to CLIP `(token:weight)` syntax. |
| `combine`            | List-extend each input's entries narrowed to its effective range. If `weight ≠ 1.0`, writes `strength = weight` into the entry's metadata dict (equivalent to a `ConditioningSetAreaStrength` on that input). No segmentation. |
| `average`            | `sum(weight_i * tokens_i for i in active)`. Same for pooled_output. Not normalized.                                                               |
| `average_normalized` | `sum(weight_i * tokens_i) / sum(weight_i)` over active inputs. Reproduces stock `ConditioningAverage` when N=2 and weights `[s, 1-s]`.            |

## How segmentation works (concat / average modes)

When inputs have different effective ranges, the node breaks the [0, 1]
timestep span at every endpoint and emits one CONDITIONING entry per
sub-interval. Only inputs whose effective range covers a sub-interval
contribute to it.

**Example:** two inputs with widget `[0,1]` and upstream
`conditioning_1 = [0.0, 0.6]`, `conditioning_2 = [0.3, 1.0]`:

| Range        | Active subset | concat output                                | average_normalized output (weights 1,1) |
|--------------|---------------|----------------------------------------------|-----------------------------------------|
| [0.0, 0.3]   | {1}           | `conditioning_1` tokens                      | `conditioning_1` tokens                 |
| [0.3, 0.6]   | {1, 2}        | `torch.cat([c1, c2], dim=1)`                 | `0.5 * c1 + 0.5 * c2`                   |
| [0.6, 1.0]   | {2}           | `conditioning_2` tokens                      | `conditioning_2` tokens                 |

## Compatibility

The node respects per-entry `start_percent`/`end_percent` via
`comfy/samplers.py:calculate_start_end_timesteps` — works with every standard
sampler (KSampler / KSamplerAdvanced / WAS_KSampler / etc.) without
modification.

Inputs may themselves be multi-entry CONDITIONING lists (e.g., output of
another instance of this node). Every entry on every connected slot is
expanded into the segmentation independently.

## Implementation notes

- Pure Python backend with no third-party dependencies beyond
  ComfyUI/PyTorch.
- A small frontend extension (`web/conditioning_merge_with_timestep_ranges.js`,
  ~120 lines) handles the dynamic-slot UX. Depends only on ComfyUI's
  `scripts/app.js` and core LiteGraph methods.
- The flexible-optional-inputs pattern is reimplemented locally (not
  imported from any third-party node pack).

## History

The first commit on `main` shipped three separate nodes (Concat / Combine /
Average) — kept for git history. The current `main` replaces them with this
single unified node. **Breaking change**: workflows referencing the old
class names (`ConditioningConcatWithTimestepRanges` etc.) must be rewired to
use `ConditioningMergeWithTimestepRanges`.

## License

MIT — see `LICENSE`.
