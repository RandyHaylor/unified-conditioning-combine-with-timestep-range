"""
Debug Conditioning — in-line passthrough that prints CONDITIONING structure to
the ComfyUI server console. Useful for verifying that start_percent /
end_percent / pooled_output / strength etc. survive each step of a chain.

Drop it anywhere on a CONDITIONING wire. Output is identical to input. Each
entry's tensor shape, dtype, device, min/max, and metadata dict keys are
printed under the given label.
"""

import torch


class DebugConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "label": ("STRING", {"default": "conditioning"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "debug"
    CATEGORY = "advanced/conditioning"

    def debug(self, conditioning, label):
        print("\n" + "=" * 80)
        print(f"DEBUG CONDITIONING: {label}")
        print(f"type: {type(conditioning)}")
        print(f"len: {len(conditioning)}")

        for i, item in enumerate(conditioning):
            print("-" * 80)
            print(f"item[{i}] type: {type(item)}")

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                cond_tensor = item[0]
                metadata = item[1]

                print(f"  tensor type: {type(cond_tensor)}")
                if isinstance(cond_tensor, torch.Tensor):
                    print(f"  tensor shape: {tuple(cond_tensor.shape)}")
                    print(f"  tensor dtype: {cond_tensor.dtype}")
                    print(f"  tensor device: {cond_tensor.device}")
                    print(f"  tensor min/max: {cond_tensor.min().item():.6f} / {cond_tensor.max().item():.6f}")

                print(f"  metadata type: {type(metadata)}")
                if isinstance(metadata, dict):
                    print("  metadata keys:")
                    for key, value in metadata.items():
                        if isinstance(value, torch.Tensor):
                            print(
                                f"    {key}: Tensor shape={tuple(value.shape)} "
                                f"dtype={value.dtype} device={value.device}"
                            )
                        else:
                            print(f"    {key}: {repr(value)}")
                else:
                    print(f"  metadata: {repr(metadata)}")
            else:
                print(f"  raw item: {repr(item)}")

        print("=" * 80 + "\n")

        return (conditioning,)
