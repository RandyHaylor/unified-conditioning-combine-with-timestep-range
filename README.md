# Conditioning Merge Nodes (with Timestep Ranges)

ComfyUI custom nodes that fix a long-standing issue with stock
`ConditioningConcat` and `ConditioningAverage`: they silently drop the
`start_percent`/`end_percent` metadata from the `from`-side input, so a
`ConditioningSetTimestepRange` placed before that input has no effect.

These nodes expose per-side timestep range widgets on both inputs, intersect
those widgets with any upstream timestep range already on the conditioning
dict, and emit a properly segmented CONDITIONING list that any standard
sampler honors.

## What's broken in stock ComfyUI

Stock `ConditioningConcat` and `ConditioningAverage` copy only `conditioning_to[i][1]`
as the output metadata dict — anything in `conditioning_from[0][1]` (including
`start_percent`/`end_percent` set by a prior `ConditioningSetTimestepRange`)
is discarded. Stock `ConditioningCombine` is fine (it list-concatenates entries).

## Nodes provided

All three appear under **advanced/conditioning**:

- **Conditioning Concat (with timestep ranges)**
- **Conditioning Combine (with timestep ranges)**
- **Conditioning Average (with timestep ranges)**

Each has start/end timestep widgets for both inputs. Widget ranges
**intersect** with upstream `start_percent`/`end_percent` already on each
entry's metadata dict, so:

- Widget `[0, 1]` is a true pass-through — upstream ranges fully control output.
- Widget can only **narrow** the upstream range, never widen it.

## How the Concat / Average segmentation works

When `to` and `from` have different effective ranges, the node breaks the
[0, 1] timestep span at every endpoint and emits one CONDITIONING entry per
sub-interval:

- Sub-intervals where only one side is active → that side's tokens.
- Sub-intervals where both are active → token-concat (or weighted blend, for
  Average).
- Sub-intervals where neither is active → nothing.

**Example:** `to = [0.0, 0.6]`, `from = [0.3, 1.0]` produces three entries:

| Range        | Content                                 |
|--------------|-----------------------------------------|
| [0.0, 0.3]   | `to` tokens only                        |
| [0.3, 0.6]   | `torch.cat(to_tokens, from_tokens, dim=1)` |
| [0.6, 1.0]   | `from` tokens only                      |

## Compatibility

| Path                                            | Result                                           |
|-------------------------------------------------|--------------------------------------------------|
| Our node → our node (any side)                  | Lossless                                         |
| Our node output → stock Concat `to` input       | Per-entry ranges preserved                       |
| Our node output → stock Concat `from` input     | Multi-entry list collapses to entry 0 (lossy)    |
| Stock Combine → our node                        | Lossless                                         |
| KSampler / KSamplerAdvanced / WAS KSampler      | All honor `start_percent`/`end_percent` natively |

Stock samplers respect the per-entry timestep ranges via
`comfy/samplers.py:calculate_start_end_timesteps`, so no special integration
is required — just drop the node in.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/RandyHaylor/unified-conditioning-combine-with-timestep-range.git conditioning_merge_nodes_with_timestep_range
```

Restart ComfyUI. The three nodes appear under `advanced/conditioning`.

## Status

This is the **initial three-node release**. A **unified single-node** version
with dynamic 1–N inputs, per-slot weights, and a four-mode dropdown
(concat / combine / average / average_normalized) is in development on
the main branch.

## License

MIT — see `LICENSE`.
