import argparse
import time

import numpy as np
import soundfile as sf
import torch
import yaml
import dotmap

from scipy.signal import stft, resample_poly
from torch_geometric.data import Data

from models.recognition import LNRecognition


# ============================================================
# Configuration
# ============================================================

SAMPLE_RATE = 16000

NUM_CHANNELS = 700

TIME_RADIUS = 20000       # microseconds = 20 ms
CHANNEL_RADIUS = 100

SKIP_CHANNELS = 10

CHECKPOINT = "checkpoints/best_model_float-v1.ckpt"


LABELS = [
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
    "neun",
]


# ============================================================
# Load FLAC
# ============================================================

def load_flac(filename):
    print()
    print("=" * 48)
    print(" FLAC INPUT")
    print("=" * 48)

    print(f"File: {filename}")

    audio, sample_rate = sf.read(
        filename,
        dtype="float32"
    )

    print(f"Original sample rate: {sample_rate} Hz")

    # --------------------------------------------------------
    # Convert stereo/multichannel to mono
    # --------------------------------------------------------

    if audio.ndim > 1:
        print(f"Channels: {audio.shape[1]}")
        audio = np.mean(audio, axis=1)
    else:
        print("Channels: 1")

    # --------------------------------------------------------
    # Resample to 16 kHz
    # --------------------------------------------------------

    if sample_rate != SAMPLE_RATE:

        print(
            f"Resampling: {sample_rate} Hz -> "
            f"{SAMPLE_RATE} Hz"
        )

        audio = resample_poly(
            audio,
            SAMPLE_RATE,
            sample_rate
        ).astype(np.float32)

        sample_rate = SAMPLE_RATE

    # --------------------------------------------------------
    # Remove DC offset
    # --------------------------------------------------------

    audio = audio - np.mean(audio)

    # --------------------------------------------------------
    # Normalize safely
    # --------------------------------------------------------

    max_value = np.max(np.abs(audio))

    if max_value > 0:
        audio = audio / max_value

    duration = len(audio) / sample_rate

    print(f"Final sample rate: {sample_rate} Hz")
    print(f"Samples: {len(audio)}")
    print(f"Duration: {duration:.3f} seconds")

    return audio, sample_rate


# ============================================================
# Audio -> Event representation
# ============================================================

def audio_to_events(audio, sample_rate):
    """
    Prototype conversion:

        FLAC waveform
              |
             STFT
              |
        log magnitude
              |
        700 frequency channels
              |
        threshold
              |
        timestamp + channel events

    NOTE:
    This is a prototype audio-to-event encoder.
    It is NOT the original neuromorphic cochlea.
    """

    print()
    print("Converting FLAC -> events...")

    # --------------------------------------------------------
    # STFT
    # --------------------------------------------------------

    nperseg = 512
    noverlap = 384

    frequencies, times, spectrum = stft(
        audio,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None
    )

    magnitude = np.abs(spectrum)

    # Avoid log(0)
    magnitude = np.maximum(
        magnitude,
        1e-10
    )

    log_magnitude = np.log1p(magnitude)

    # --------------------------------------------------------
    # Map STFT frequency bins to 700 channels
    # --------------------------------------------------------

    num_freq_bins = log_magnitude.shape[0]

    channel_energy = np.zeros(
        (NUM_CHANNELS, log_magnitude.shape[1]),
        dtype=np.float32
    )

    source_positions = np.linspace(
        0,
        num_freq_bins - 1,
        NUM_CHANNELS
    )

    for channel in range(NUM_CHANNELS):

        source_index = int(
            round(source_positions[channel])
        )

        source_index = min(
            source_index,
            num_freq_bins - 1
        )

        channel_energy[channel, :] = (
            log_magnitude[source_index, :]
        )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = np.percentile(
        channel_energy,
        85
    )

    event_mask = channel_energy > threshold

    channel_indices, time_indices = np.where(
        event_mask
    )

    if len(channel_indices) == 0:
        raise RuntimeError(
            "No events were generated from this audio file. "
            "Try an audio file containing speech or reduce "
            "the threshold."
        )

    # --------------------------------------------------------
    # Convert timestamps to microseconds
    # --------------------------------------------------------

    event_times_us = (
        times[time_indices] * 1_000_000.0
    )

    event_channels = channel_indices.astype(
        np.int32
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    order = np.argsort(event_times_us)

    event_times_us = event_times_us[order]

    event_channels = event_channels[order]

    events = np.column_stack(
        (
            event_times_us,
            event_channels
        )
    )

    print(f"Generated events: {len(events)}")

    print(
        f"Active channels: "
        f"{len(np.unique(event_channels))}"
    )

    print(
        f"Time range: "
        f"{event_times_us.min():.0f} - "
        f"{event_times_us.max():.0f} us"
    )

    return events


# ============================================================
# Graph construction
# ============================================================

def events_to_graph(events):
    """
    Convert:

        [timestamp_us, channel]

    into a PyTorch Geometric graph.

    IMPORTANT:

    The trained GNN's conv1 expects 4 inputs.

    MyPointNetConv internally creates:

        pos_j - pos_i

    which gives 2 positional features.

    Therefore x must contain only 2 node features:

        x[:,0] = normalized time
        x[:,1] = normalized channel

    Total:

        2 node features + 2 positional features = 4
    """

    print()
    print("Building graph...")

    num_events = len(events)

    if num_events == 0:
        raise RuntimeError(
            "No events available for graph construction."
        )

    # --------------------------------------------------------
    # Event positions
    #
    # Model expects normalized positions.
    # --------------------------------------------------------

    max_time = np.max(events[:, 0])

    if max_time <= 0:
        max_time = 1.0

    pos = np.zeros(
        (num_events, 2),
        dtype=np.float32
    )

    pos[:, 0] = (
        events[:, 0] / max_time
    )

    pos[:, 1] = (
        events[:, 1] / NUM_CHANNELS
    )

    # --------------------------------------------------------
    # Node features
    #
    # ONLY TWO FEATURES.
    #
    # MyPointNetConv adds:
    #     pos_j - pos_i
    #
    # internally.
    # --------------------------------------------------------

    x = np.zeros(
        (num_events, 2),
        dtype=np.float32
    )

    x[:, 0] = pos[:, 0]
    x[:, 1] = pos[:, 1]

    # --------------------------------------------------------
    # Build directed graph
    #
    # For each event, connect to the most recent event on
    # neighboring channels.
    # --------------------------------------------------------

    channel_last_event = [
        None
    ] * NUM_CHANNELS

    edge_sources = []
    edge_targets = []

    for i in range(num_events):

        event_time = events[i, 0]

        event_channel = int(
            events[i, 1]
        )

        # ----------------------------------------------------
        # Search neighboring channels
        # ----------------------------------------------------

        channel_min = max(
            0,
            event_channel -
            CHANNEL_RADIUS * SKIP_CHANNELS
        )

        channel_max = min(
            NUM_CHANNELS,
            event_channel +
            CHANNEL_RADIUS * SKIP_CHANNELS +
            1
        )

        for neighbor_channel in range(
            channel_min,
            channel_max
        ):

            if neighbor_channel == event_channel:
                continue

            previous = channel_last_event[
                neighbor_channel
            ]

            if previous is None:
                continue

            neighbor_index, neighbor_time = previous

            # Time difference in microseconds
            time_difference = (
                event_time -
                neighbor_time
            )

            if (
                time_difference >= 0
                and
                time_difference <= TIME_RADIUS
            ):
                edge_sources.append(
                    neighbor_index
                )

                edge_targets.append(i)

        # ----------------------------------------------------
        # Current event becomes the latest event on channel
        # ----------------------------------------------------

        channel_last_event[
            event_channel
        ] = (
            i,
            event_time
        )

    # --------------------------------------------------------
    # PyTorch tensors
    # --------------------------------------------------------

    x = torch.from_numpy(x).float()

    pos = torch.from_numpy(pos).float()

    if len(edge_sources) == 0:

        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    else:

        edge_index = torch.tensor(
            [
                edge_sources,
                edge_targets
            ],
            dtype=torch.long
        )

    graph = Data(
        x=x,
        pos=pos,
        edge_index=edge_index
    )

    print(f"Nodes : {graph.num_nodes}")
    print(f"Edges : {graph.num_edges}")

    print(
        f"Node feature shape: "
        f"{tuple(graph.x.shape)}"
    )

    print(
        f"Position shape: "
        f"{tuple(graph.pos.shape)}"
    )

    print(
        f"Edge shape: "
        f"{tuple(graph.edge_index.shape)}"
    )

    return graph


# ============================================================
# Load trained model
# ============================================================

def load_model(checkpoint):
    print()
    print("Loading GNN...")

    # --------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------

    with open(
        "configs/digits.yaml",
        "r"
    ) as f:

        cfg = yaml.load(
            f,
            Loader=yaml.FullLoader
        )

    cfg = dotmap.DotMap(cfg)

    # --------------------------------------------------------
    # Load Lightning checkpoint
    #
    # weights_only=False is required for this trusted
    # checkpoint because it contains DotMap configuration
    # information.
    # --------------------------------------------------------

    model = LNRecognition.load_from_checkpoint(
        checkpoint,
        weights_only=False
    )

    # --------------------------------------------------------
    # Actual GCN
    # --------------------------------------------------------

    gnn = model.model

    gnn.eval()

    for param in gnn.parameters():
        param.requires_grad = False

    print("Model loaded.")

    print(gnn)

    return gnn


# ============================================================
# Classification
# ============================================================

def classify(model, graph):
    print()
    print("Running GNN inference...")

    # --------------------------------------------------------
    # CPU inference
    # --------------------------------------------------------

    graph = graph.to("cpu")

    model = model.to("cpu")

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    start_time = time.perf_counter()

    with torch.no_grad():

        output, _ = model(graph)

    inference_time = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------------------
    # Convert logits to probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        output,
        dim=-1
    )

    probabilities = probabilities.squeeze(0)

    # --------------------------------------------------------
    # Top prediction
    # --------------------------------------------------------

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    confidence = float(
        probabilities[predicted_index].item()
    )

    predicted_label = LABELS[
        predicted_index
    ]

    # --------------------------------------------------------
    # Top 5
    # --------------------------------------------------------

    top_k = min(
        5,
        len(LABELS)
    )

    top_values, top_indices = torch.topk(
        probabilities,
        top_k
    )

    print()
    print("=" * 48)
    print(" CLASSIFICATION RESULT")
    print("=" * 48)

    print(
        f"Prediction : {predicted_label}"
    )

    print(
        f"Class      : {predicted_index}"
    )

    print(
        f"Confidence : "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Inference time : "
        f"{inference_time:.3f} seconds"
    )

    print()
    print("Top 5 predictions:")

    for rank in range(top_k):

        index = int(
            top_indices[rank].item()
        )

        probability = float(
            top_values[rank].item()
        )

        print(
            f"{rank + 1}. "
            f"{LABELS[index]:<8} "
            f"{probability * 100:6.2f}%"
        )

    print("=" * 48)

    return predicted_label, confidence


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Classify an external FLAC audio file "
            "using the trained GNN."
        )
    )

    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the .flac file"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=CHECKPOINT,
        help=(
            "GNN checkpoint "
            "(default: checkpoints/best_model_float-v1.ckpt)"
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Check extension
    # --------------------------------------------------------

    if not args.audio_file.lower().endswith(
        ".flac"
    ):

        print(
            "WARNING: Input file does not have a "
            ".flac extension."
        )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sample_rate = load_flac(
        args.audio_file
    )

    # --------------------------------------------------------
    # Audio -> events
    # --------------------------------------------------------

    events = audio_to_events(
        audio,
        sample_rate
    )

    # --------------------------------------------------------
    # Events -> graph
    # --------------------------------------------------------

    graph = events_to_graph(
        events
    )

    # --------------------------------------------------------
    # Load GNN
    # --------------------------------------------------------

    model = load_model(
        args.checkpoint
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    classify(
        model,
        graph
    )


if __name__ == "__main__":
    main()