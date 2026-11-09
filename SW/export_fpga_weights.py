
from pathlib import Path

import torch
import yaml
import dotmap

from models.recognition import LNRecognition


CHECKPOINT = "checkpoints/best_model_calibrated.ckpt"
OUTPUT_DIR = Path("fpga_export_calibrated")


def checked_int(value, name):
    value = float(value)
    rounded = round(value)

    if abs(value - rounded) > 0.001:
        raise ValueError(f"{name} is not an integer: {value}")

    return int(rounded)


def encode_signed32(value):
    if not -(2**31) <= value < 2**31:
        raise ValueError(f"32-bit bias overflow: {value}")

    return value & 0xFFFFFFFF


def export_layer(conv, layer_number):
    bits = 16 if layer_number == 1 else 8
    input_dim = 4 if layer_number == 1 else 66

    centered = conv.qlinear.weight.detach().cpu()
    biases = conv.qlinear.bias.detach().cpu()

    zero_point = checked_int(
        conv.observer_weight.zero_point.item(),
        "weight zero point"
    )

    assert tuple(centered.shape) == (64, input_dim)
    assert tuple(biases.shape) == (64,)

    max_weight = (1 << bits) - 1
    words = []

    for channel in range(64):
        raw_weights = []

        for j in range(input_dim):
            centered_value = checked_int(
                centered[channel, j].item(),
                f"conv{layer_number} channel{channel} weight{j}"
            )

            raw = centered_value + zero_point

            if not 0 <= raw <= max_weight:
                raise ValueError(
                    f"Unsigned weight overflow: "
                    f"layer={layer_number}, channel={channel}, "
                    f"weight={j}, value={raw}"
                )

            raw_weights.append(raw)

        bias = checked_int(
            biases[channel].item(),
            f"conv{layer_number} channel{channel} bias"
        )

        bias_encoded = encode_signed32(bias)

        if layer_number == 1:
            # Bits 31:0 = bias
            # Bits 95:32 = four 16-bit weights
            packed = bias_encoded

            for j, weight in enumerate(raw_weights):
                packed |= weight << (32 + 16 * j)

            words.append(f"{packed:024x}")

            # Verify the packed representation.
            assert (packed & 0xFFFFFFFF) == bias_encoded

            for j, weight in enumerate(raw_weights):
                decoded = (packed >> (32 + 16*j)) & 0xFFFF
                assert decoded == weight

        else:
            # First eight words: 64 weights, 8 per word.
            channel_words = []

            for block in range(8):
                packed = 0

                for j in range(8):
                    weight = raw_weights[block * 8 + j]
                    packed |= weight << (8 * j)

                channel_words.append(packed)

            # Final word:
            # bits 31:0   = signed bias
            # bits 47:32  = reserved
            # bits 55:48  = reserved
            # bits 63:56  = remaining weights (see note below)
            #
            # The RTL actually reads:
            # bias = [543:512]
            # weight64 = [567:560]
            # weight65 = [575:568]
            #
            # Relative to word 8:
            # bias = [31:0]
            # weight64 = [55:48]
            # weight65 = [63:56]

            final_word = bias_encoded
            final_word |= raw_weights[64] << 48
            final_word |= raw_weights[65] << 56

            channel_words.append(final_word)

            assert len(channel_words) == 9

            # Round-trip verification.
            reconstructed = []

            for word in channel_words[:8]:
                for j in range(8):
                    reconstructed.append((word >> (8*j)) & 0xFF)

            reconstructed.append(
                (channel_words[8] >> 48) & 0xFF
            )
            reconstructed.append(
                (channel_words[8] >> 56) & 0xFF
            )

            assert reconstructed == raw_weights
            assert (channel_words[8] & 0xFFFFFFFF) == bias_encoded

            words.extend(f"{word:016x}" for word in channel_words)

    expected = 64 if layer_number == 1 else 576
    assert len(words) == expected

    output_file = OUTPUT_DIR / f"conv_{layer_number}_weights.mem"
    output_file.write_text("\n".join(words) + "\n")

    return {
        "layer": layer_number,
        "words": len(words),
        "weight_zero_point": zero_point,
        "input_zero_point": checked_int(
            conv.observer_input.zero_point.item(),
            "input zero point"
        ),
        "output_zero_point": checked_int(
            conv.observer_output.zero_point.item(),
            "output zero point"
        ),
        "multiplier": checked_int(
            conv.qscale_m.item(),
            "fixed-point multiplier"
        ),
    }


def main():
    with open("configs/digits.yaml", "r") as f:
        cfg = dotmap.DotMap(yaml.safe_load(f))

    model = LNRecognition.load_from_checkpoint(
        CHECKPOINT,
        config=cfg,
        map_location="cpu",
        weights_only=False
    )

    model.eval()

    with torch.no_grad():
        model.model.quantize()

    OUTPUT_DIR.mkdir(exist_ok=True)

    report = []

    for i in range(1, 5):
        conv = getattr(model.model, f"conv{i}")
        info = export_layer(conv, i)
        report.append(info)

        print(
            f"Conv {i}: exported {info['words']} words "
            f"to {OUTPUT_DIR}"
        )

    print("\nQuantization parameters:")
    for info in report:
        print(info)

    print("\nExport completed.")


if __name__ == "__main__":
    main()
