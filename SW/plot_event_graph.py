import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# Make the compiled C++ extension importable.
sys.path.insert(0, str(Path("data").resolve()))
import edge_generator

# --------------------------------------------------
# 1. Load the real audio spike events
# --------------------------------------------------

spike_file = (
    "spike_data_generator/outputs/"
    "lang-german_speaker-11_trial-9_digit-9_spikes.npz"
)

with np.load(spike_file) as f:
    spikes = f["arr_0"]

times = spikes[0].astype(np.float32)
channels = spikes[1].astype(np.float32)

# --------------------------------------------------
# 2. Load project configuration
# --------------------------------------------------

with open("configs/digits.yaml") as f:
    config = yaml.safe_load(f)

print("Configuration sections:", list(config.keys()))

# These parameters must match the constructor in
# data/dataset.py. Verify their locations in YAML.
general = config["general"]

num_channels = int(general["num_channels"])

# Adjust these paths if dataset.py shows a different
# configuration section or field name.
graph = config.get("graph", general)

channel_radius = float(graph["channel_radius"])
time_radius = float(graph["time_radius"])
# time_window = float(graph["time_window"])
time_window = float(
    graph.get(
        "time_window",
        config.get("preprocess", {}).get("time_window",
            config.get("general", {}).get("time_window", 0))
    )
)
skip_channels = int(graph["skip_channels"])
features = str(graph["features"])

# --------------------------------------------------
# 3. Generate the actual event graph
# --------------------------------------------------

# The C++ extension accesses the tensors with
# idx * 2, so supply interleaved (N,2) data
# and pass each column as a strided view.
pos = torch.tensor(
    np.column_stack((times, channels)),
    dtype=torch.float32
)

generator = edge_generator.EdgeGenerator(
    num_channels,
    channel_radius,
    time_radius,
    time_window,
    skip_channels,
    features
)

edge_index, node_features = generator.generate_edges(
    pos[:, 0],
    pos[:, 1]
)

edges = edge_index.detach().cpu().numpy()

print("Total nodes:", len(times))
print("Total edges:", edges.shape[1])
print("Node features:", tuple(node_features.shape))

# --------------------------------------------------
# 4. Select a short time window for visualization
# --------------------------------------------------

start_ms = 100
end_ms = 150

time_ms = times * 1000

selected = (
    (time_ms >= start_ms) &
    (time_ms <= end_ms)
)

# Retain only edges with both endpoints in the window.
source = edges[0]
target = edges[1]

edge_mask = selected[source] & selected[target]

source = source[edge_mask]
target = target[edge_mask]

segments = np.stack(
    [
        np.column_stack((time_ms[source], channels[source])),
        np.column_stack((time_ms[target], channels[target]))
    ],
    axis=1
) if len(source) else np.empty((0, 2, 2))

# --------------------------------------------------
# 5. Plot the graph
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))

if len(segments):
    lines = LineCollection(
        segments,
        colors="steelblue",
        linewidths=0.5,
        alpha=0.25
    )
    ax.add_collection(lines)

ax.scatter(
    time_ms[selected],
    channels[selected],
    s=8,
    c="darkred",
    zorder=3
)

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Frequency Channel")
ax.set_title("Spectro-Temporal Event Graph - German Digit 9")

ax.set_xlim(start_ms, end_ms)
ax.set_ylim(0, num_channels)
ax.grid(alpha=0.2)

fig.tight_layout()

output = Path("results/figures/event_graph.png")
fig.savefig(output, dpi=300, bbox_inches="tight")

plt.close(fig)

print("Nodes displayed:", int(selected.sum()))
print("Edges displayed:", len(source))
print("Saved:", output.resolve())

