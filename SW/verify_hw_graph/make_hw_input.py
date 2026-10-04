import os
import torch
import numpy as np
import json
from utils import GraphDataset_new
from datetime import datetime

# Initialize the dataset
test_ds = GraphDataset_new("/bignobackup/hn280727/event-audio/graph_hw_modified/e_to_n-tr-0.02-cr-hemi/deg_10/train", augm=[], mode="label_shd")
# test_ds = GraphDataset_new("/bignobackup/hn280727/event-audio/graphs_new/e_to_n-1-tr-0.02-cr-10-hemi-True-sample-uniform/deg_10/test", augm=[], mode="label_shd")
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

# Output file paths
os.makedirs(f"hw_input/{dt_string}", exist_ok=True)
bit_output_file = f"hw_input/{dt_string}/bit_spike_data.txt"
row_output_file = f"hw_input/{dt_string}/row_spike_data.txt"
edges_output_file = f"hw_input/{dt_string}/formatted_edges_data.json"

# Extract pos_item data
t = test_ds[1].x[:, 0] * (2**20)  # Time information
f = test_ds[1].x[:, 1]  # Frequency information
edges = test_ds[1].edge_index  # Edge information

# Convert pos_item to numpy arrays
t_np = t.numpy()
f_np = f.numpy()

# Output pos_item in binary and numeric formats
with open(bit_output_file, 'w') as bit_file, open(row_output_file, 'w') as row_file:
    for t_val, f_val in zip(t_np, f_np):
        time = int(t_val)
        freq = int(f_val)
        time_bits = format(time, '021b')  # Convert time to 21-bit binary
        freq_bits = format(freq, '021b')  # Convert frequency to 21-bit binary
        bit_file.write(f"{time_bits}{freq_bits}\n")
        row_file.write(f"{time} {freq}\n")

# Prepare edge data in JSON format
edges_np = edges.numpy()
json_output = []

# Process each event (index in t_np and f_np)
for event in range(len(t_np)):
    t_event = int(t_np[event])  # Event time
    f_event = int(f_np[event])  # Event frequency

    # Add event and edges to JSON output
    json_entry = {
        "pos_item": {"t": t_event, "f": f_event},
        "edges": [],
        "average_edge": {"avg_t": 0, "avg_f": 0}  # Initialize averages to 0
    }

    # Find edges associated with the current event
    edge_indices = np.where(edges_np[0] == event)[0]
    t_edge_sum = 0
    f_edge_sum = 0
    edge_count = len(edge_indices)

    for edge_idx in edge_indices:
        target = edges_np[1][edge_idx]
        t_target = int(t_np[target])  # Target time
        f_target = int(f_np[target])  # Target frequency
        t_diff = round(abs(t_target - t_event))  # Time difference
        f_diff = round(abs(f_target - f_event))  # Frequency difference

        # Add to sum for averaging
        t_edge_sum += t_target
        f_edge_sum += f_target

        # Append edge details
        edge_entry = {"t": t_target, "f": f_target, "dt": t_diff, "df": f_diff}
        json_entry["edges"].append(edge_entry)

    # Calculate averages if there are edges
    if edge_count > 0:
        json_entry["average_edge"]["avg_t"] = t_edge_sum / edge_count
        json_entry["average_edge"]["avg_f"] = f_edge_sum / edge_count

    json_output.append(json_entry)

# Save JSON file with correct formatting
with open(edges_output_file, 'w') as output_file:
    json.dump(json_output, output_file, indent=4)
