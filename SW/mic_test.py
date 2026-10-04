import argparse
import numpy as np
import torch
import sounddevice as sd
from scipy.signal import stft
import edge_generator

from torch_geometric.data import Data

from models.recognition import LNRecognition


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000
RECORD_SECONDS = 1.0

NUM_CHANNELS = 700

TIME_RADIUS = 20000       # microseconds
CHANNEL_RADIUS = 100
SKIP_CHANNELS = 10

CHECKPOINT = "checkpoints/best_model_float-v1.ckpt"
MIN_AUDIO_RMS = 0.01  # About -40 dBFS; suppresses low-level room/mic noise.


# ============================================================
# 1. Record microphone
# ============================================================

def record_microphone(seconds=1.0, sample_rate=16000):

    print()
    print("================================================")
    print(" MICROPHONE")
    print("================================================")
    print("Speak now...")
    
    audio = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio[:, 0]

    print("Recording finished.")
    print(f"Samples: {len(audio)}")
    print(f"Duration: {len(audio) / sample_rate:.3f} s")

    return audio


# ============================================================
# 2. Convert microphone audio to events
#
# IMPORTANT:
# This is a prototype encoder.
# It is NOT the original SHD cochlea encoder.
# ============================================================

def audio_to_events(
    audio,
    sample_rate=16000,
    num_channels=700,
    threshold_percentile=85,
    min_rms=MIN_AUDIO_RMS,
):

    print()
    print("Converting audio -> events...")

    audio = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    print(f"Audio RMS: {rms:.5f} (minimum: {min_rms:.5f})")
    if rms < min_rms:
        print("No speech detected: recording level is too low.")
        return np.empty((0, 2), dtype=np.float32)

    # STFT
    frequencies, times, Z = stft(
        audio,
        fs=sample_rate,
        nperseg=512,
        noverlap=384,
        boundary=None
    )

    magnitude = np.abs(Z)

    # Only positive-frequency bins
    magnitude = magnitude + 1e-12

    # Log magnitude
    log_mag = np.log(magnitude)

    # Map frequency bins onto 700 channels
    channel_indices = np.linspace(
        0,
        len(frequencies) - 1,
        num_channels
    ).astype(int)

    channel_energy = log_mag[channel_indices, :]

    # Global threshold
    threshold = np.percentile(
        channel_energy,
        threshold_percentile
    )

    events = []

    for ch in range(num_channels):

        signal = channel_energy[ch]

        for frame, value in enumerate(signal):

            if value > threshold:

                # STFT time -> microseconds
                timestamp_us = int(
                    times[frame] * 1_000_000
                )

                events.append(
                    (timestamp_us, ch)
                )

    if len(events) == 0:
        print("WARNING: No events generated.")
        return np.empty((0, 2), dtype=np.float32)

    events = np.asarray(events, dtype=np.float32)

    # Sort by timestamp
    events = events[
        np.argsort(events[:, 0])
    ]

    print(f"Generated events: {len(events)}")
    print(
        f"Active channels: "
        f"{len(np.unique(events[:, 1]))}"
    )

    return events


# ============================================================
# 3. Construct graph
#
# Similar concept to repository's graph generation:
# event -> node
# previous nearby event -> edge
# ============================================================

def events_to_graph(
    events,
    time_radius=TIME_RADIUS,
    channel_radius=CHANNEL_RADIUS,
    num_channels=NUM_CHANNELS,
    skip_channels=SKIP_CHANNELS,
    time_window=1_000_000,
    features="global",
):

    if len(events) == 0:
        return None

    print()
    print("Building graph...")

    num_events = len(events)

    # Match the preprocessing used by SpikingDS: timestamps are relative
    # microseconds at this point, and EdgeGenerator computes both edges and
    # the model's configured two-column node features.
    events = events.copy()
    events[:, 0] -= events[0, 0]
    events = events[events[:, 0] < time_window]
    if len(events) == 0:
        return None
    num_events = len(events)

    # --------------------------------------------------------
    # Node positions
    #
    # pos[:,0] = normalized time
    # pos[:,1] = normalized channel
    # --------------------------------------------------------

    pos = torch.tensor(
        np.column_stack([
            events[:, 0] / float(time_window),
            events[:, 1] / float(num_channels)
        ]),
        dtype=torch.float32
    )
    edge_gen = edge_generator.EdgeGenerator(
        num_channels, channel_radius, time_radius, time_window,
        skip_channels, features
    )
    event_tensor = torch.from_numpy(events.astype(np.float32))
    edge_index, x = edge_gen.generate_edges(
        event_tensor[:, 0],
        event_tensor[:, 1],
    )
    edge_index = edge_index.to(dtype=torch.long)

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index
    )

    print(f"Nodes : {data.num_nodes}")
    print(f"Edges : {data.num_edges}")

    return data


# ============================================================
# 4. Load GNN
# ============================================================

def load_model(checkpoint):

    print()
    print("Loading GNN...")

    model = LNRecognition.load_from_checkpoint(
        checkpoint,
        weights_only=False
    )

    model.model.eval()

    for param in model.model.parameters():
        param.requires_grad = False

    print("Model loaded.")
    print(model.model)

    return model.model


# ============================================================
# 5. Run inference
# ============================================================

def classify(model, graph):

    print()
    print("Running GNN inference...")

    with torch.no_grad():
        output, _ = model(graph)

        probabilities = torch.softmax(
            output,
            dim=-1
        )

        prediction = torch.argmax(
            probabilities,
            dim=-1
        ).item()

    labels = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "null",
        "eins",
        "zwei",
        "drei",
        "vier",
        "fuenf",
        "sechs",
        "sieben",
        "acht",
        "neun"
    ]

    print()
    print("================================================")
    print(" RESULT")
    print("================================================")
    print(f"Class index : {prediction}")
    label = labels[prediction] if prediction < len(labels) else f"class_{prediction}"
    print(f"Prediction  : {label}")
    print(
        f"Confidence  : "
        f"{probabilities[0, prediction].item() * 100:.2f}%"
    )

    print()
    print("Top predictions:")

    values, indices = torch.topk(
        probabilities[0],
        k=5
    )

    for value, index in zip(values, indices):

        print(
            f"  {(labels[index.item()] if index.item() < len(labels) else f'class_{index.item()}'):8s} "
            f"{value.item() * 100:6.2f}%"
        )

    return prediction


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seconds",
        type=float,
        default=RECORD_SECONDS
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT
    )

    parser.add_argument(
        "--min-rms",
        type=float,
        default=MIN_AUDIO_RMS,
        help="Minimum recording RMS for inference (lower for a quiet microphone)."
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Record
    # --------------------------------------------------------

    audio = record_microphone(
        seconds=args.seconds,
        sample_rate=SAMPLE_RATE
    )

    # --------------------------------------------------------
    # Audio -> events
    # --------------------------------------------------------

    events = audio_to_events(
        audio,
        sample_rate=SAMPLE_RATE,
        min_rms=args.min_rms,
    )

    if len(events) == 0:
        print("No usable speech events. Try speaking clearly or lower --min-rms if your microphone level is low.")
        return

    # --------------------------------------------------------
    # Events -> graph
    # --------------------------------------------------------

    graph = events_to_graph(events)

    if graph is None:
        print("Could not create graph.")
        return

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        args.checkpoint
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    classify(
        model,
        graph
    )


if __name__ == "__main__":
    main()
