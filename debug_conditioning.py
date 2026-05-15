"""
Debug Conditioning — in-line passthrough that prints CONDITIONING structure to
the ComfyUI server console AND returns the same dump as a STRING output so it
can be wired into a ShowText / SaveText / similar node for inspection without
tailing the server console.

Drop it anywhere on a CONDITIONING wire. The CONDITIONING output is identical
to the input. The debug_text output is the same multi-line dump string that
was printed.
"""

import io
import torch


def _build_debug_dump_string_for_conditioning(conditioning_value, label_text):
    output_string_io_buffer = io.StringIO()

    def write_line(text_to_write):
        output_string_io_buffer.write(text_to_write + "\n")

    write_line("=" * 80)
    write_line(f"DEBUG CONDITIONING: {label_text}")
    write_line(f"type: {type(conditioning_value)}")
    write_line(f"len: {len(conditioning_value)}")

    for i, item in enumerate(conditioning_value):
        write_line("-" * 80)
        write_line(f"item[{i}] type: {type(item)}")

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            cond_tensor = item[0]
            metadata = item[1]

            write_line(f"  tensor type: {type(cond_tensor)}")
            if isinstance(cond_tensor, torch.Tensor):
                write_line(f"  tensor shape: {tuple(cond_tensor.shape)}")
                write_line(f"  tensor dtype: {cond_tensor.dtype}")
                write_line(f"  tensor device: {cond_tensor.device}")
                write_line(f"  tensor min/max: {cond_tensor.min().item():.6f} / {cond_tensor.max().item():.6f}")

            write_line(f"  metadata type: {type(metadata)}")
            if isinstance(metadata, dict):
                write_line("  metadata keys:")
                for key, value in metadata.items():
                    if isinstance(value, torch.Tensor):
                        write_line(
                            f"    {key}: Tensor shape={tuple(value.shape)} "
                            f"dtype={value.dtype} device={value.device}"
                        )
                    else:
                        write_line(f"    {key}: {repr(value)}")
            else:
                write_line(f"  metadata: {repr(metadata)}")
        else:
            write_line(f"  raw item: {repr(item)}")

    write_line("=" * 80)
    return output_string_io_buffer.getvalue()


class DebugConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "label": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "debug_text")
    FUNCTION = "debug"
    CATEGORY = "advanced/conditioning"
    OUTPUT_NODE = True

    def debug(self, conditioning, label):
        debug_dump_string = _build_debug_dump_string_for_conditioning(conditioning, label)
        print("\n" + debug_dump_string + "\n")
        return (conditioning, debug_dump_string)
