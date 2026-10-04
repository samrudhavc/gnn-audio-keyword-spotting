import os
import h5py
import torch
import numpy as np
import networkx as nx
import warnings
from utils import SpikeTrainList
from argparse import ArgumentParser
from torch_geometric.utils import from_networkx
from scipy.spatial.distance import pdist, squareform

NUM_CHANNELS = 700
warnings.filterwarnings("ignore")
SKIP_CHANNELS = 10

parser = ArgumentParser(description="Graph generation for spike data")
parser.add_argument("--mode", type=str, default="label_shd", help="Mode of operation")
parser.add_argument("--out_dir", type=str, default="/bignobackup/hn280727/event-audio/graph_hw_modified", help="Output directory")
parser.add_argument("--raw_data", type=str, default="/bignobackup/hn280727/event-audio/hdspikes", help="Data folder path")
parser.add_argument("--time_radius", type=float, default=0.02, help="Time radius (in seconds)")
parser.add_argument("--neighbor_channels", type=str, default="10", help="Comma-separated list of neighbor channels")
parser.add_argument("--train_valid_test", type=str, default="all", help="Dataset split (train, test, valid, all)")

args = parser.parse_args()
neighbor_channels_list = list(map(int, args.neighbor_channels.split(",")))

if "train" in args.train_valid_test or args.train_valid_test == "all":
    train_file = h5py.File(
        f'{args.raw_data}/{"shd" if "shd" in args.mode else "ssc"}_train.h5', "r"
    )
    x_train_data = train_file["spikes"]
    x_train_data = SpikeTrainList(x_train_data).get_data()
    y_train_labels = torch.tensor(np.array(train_file["labels"], dtype=np.compat.long))
if "test" in args.train_valid_test or args.train_valid_test == "all":
    test_file = h5py.File(
        f'{args.raw_data}/{"shd" if "shd" in args.mode else "ssc"}_test.h5', "r"
    )
    x_test_data = test_file["spikes"]
    y_test_labels = torch.tensor(np.array(test_file["labels"], dtype=np.compat.long))
    x_test_data = SpikeTrainList(x_test_data).get_data()
if (
    ("valid" in args.train_valid_test or args.train_valid_test == "all")
    and "ssc" in args.mode
):
    valid_file = h5py.File(
        f'{args.raw_data}/{"shd" if "shd" in args.mode else "ssc"}_valid.h5', "r"
    )
    x_valid_data = valid_file["spikes"]
    y_valid_labels = torch.tensor(np.array(valid_file["labels"], dtype=np.compat.long))
    x_valid_data = SpikeTrainList(x_valid_data).get_data()
    
    
def create_digraph_from_points(sparse_spikes, time_radius, channel_radius):
    n = len(sparse_spikes)
    
    if n == 0:
        return None  

    sparse_spikes = sparse_spikes.numpy()
    # Generate edges
    edges = []
    channel_last_event = [None] * NUM_CHANNELS

    for i in range(n):
        event_time = sparse_spikes[i][0]  # Event time
        event_channel = int(sparse_spikes[i][1])  # Event channel (integerized)
        channel_last_event[event_channel] = (i, event_time)
        # for neighbor_channel in range(
        #     max(0, event_channel - channel_radius*SKIP_CHANNELS),
        #     min(NUM_CHANNELS, event_channel + channel_radius*SKIP_CHANNELS + 1),
        #     SKIP_CHANNELS
        # ):
        for neighbor_channel in range(
            event_channel - channel_radius*SKIP_CHANNELS,
            event_channel + channel_radius*SKIP_CHANNELS + 1,
            SKIP_CHANNELS
        ):
            if neighbor_channel < 0 or neighbor_channel >= NUM_CHANNELS:
                continue
            if neighbor_channel != event_channel:
                if channel_last_event[neighbor_channel] is not None:
                    neighbor_index, neighbor_time = channel_last_event[neighbor_channel]
                    
                    if abs(event_time - neighbor_time) <= time_radius and neighbor_time <= event_time:
                        # edges.append((neighbor_index, i))
                        edges.append((i,neighbor_index))

    # Generate directed graph and set node features
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(range(n))
    directed_graph.add_edges_from(edges)

    # Convert to torch_geometric format
    data = from_networkx(directed_graph)
    data.x = torch.from_numpy(sparse_spikes).float()
    
    # return data
    return [data]   


def generate_digraphs_for_dataset(spike_data, labels, args, dataset_type="train", params="",neighbor_channels=10):
    # print(neighbor_channels)
    for index, spike_sample in enumerate(spike_data):
            generated_graphs = create_digraph_from_points(spike_sample, args.time_radius, neighbor_channels)
            for id_degree, graph in enumerate(generated_graphs):
                graph.y = labels[index].type(torch.LongTensor)
                output_dir = f"{args.out_dir}/{params}/deg_{neighbor_channels}/{dataset_type}"
                os.makedirs(output_dir, exist_ok=True)
                torch.save(
                    graph,
                    f"{output_dir}/sample{index}.pt",
                )
                print(f"Progress: {index + 1} / {len(spike_data)} (neighbor_channels: {neighbor_channels})", end="\r")

PARAMS = "-".join(["e_to_n", "tr", str(args.time_radius), "cr", "hemi"])

try:
    os.makedirs(f"{args.out_dir}/{PARAMS}", exist_ok=True)
except Exception:
    print("Failed to create path")

for neighbor_channels in neighbor_channels_list:
    try:
        os.makedirs(
            f"{args.out_dir}/{PARAMS}/deg_{neighbor_channels}/train", exist_ok=True
        )
        os.makedirs(
            f"{args.out_dir}/{PARAMS}/deg_{neighbor_channels}/test", exist_ok=True
        )
        warnings.filterwarnings("ignore")
    except Exception:
        print("Failed to create path")

# Generate graphs for training data
if "train" in args.train_valid_test or args.train_valid_test == "all":
    for neighbor_channels in neighbor_channels_list:
        generate_digraphs_for_dataset(x_train_data, y_train_labels, args, dataset_type="train", params=PARAMS,neighbor_channels=neighbor_channels)

# Generate graphs for test data
if "test" in args.train_valid_test or args.train_valid_test == "all":
    for neighbor_channels in neighbor_channels_list:
        generate_digraphs_for_dataset(x_test_data, y_test_labels, args, dataset_type="test", params=PARAMS,neighbor_channels=neighbor_channels)
