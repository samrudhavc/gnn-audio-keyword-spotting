import numpy as np
import torch

import edge_generator

def generate_edges(times: torch.Tensor, 
                    channels: torch.Tensor,
                    features: str = 'global',
                    time_radius: float = 0.02,
                    time_window: float = 1,
                    channel_radius: int = 100,
                    skip_channels: int = 10,
                    num_channels: int = 700):
    edges = []
    feature = []
    
    channel_last_event = [None] * 700

    for idx, (time, channel) in enumerate(zip(times, channels)):
        
        if time == 62378 or time==48615 or time==48614:
            print(time, channel)
            pass

        sum_t = 0
        sum_channel = 0
        sum_idx = 0

        time, channel = time.item(), channel.item()

        for n_channel in range(int(channel - channel_radius), 
                                int(channel + channel_radius + 1), 
                                skip_channels):

            if n_channel < 0 or n_channel >= num_channels:
                continue
            
            if channel_last_event[n_channel] is not None:
                n_time, n_idx = channel_last_event[n_channel]

                if time - n_time <= time_radius:
                    edges.append((n_idx, idx))

                    if features == 'local':
                        sum_t += (time - n_time)
                        sum_channel += (channel - n_channel)
                    
                    elif features == 'global':
                        sum_t += n_time
                        sum_channel += n_channel

                    sum_idx += 1

        if sum_idx == 0:
            mean_t = 0
            mean_channel = 0
        else:
            mean_t = sum_t / sum_idx
            mean_t = round(mean_t + 1e-6) # Add 1e-6 because Python is stupid
            mean_channel = sum_channel / sum_idx
            mean_channel = round(mean_channel + 1e-6)

        channel_last_event[int(channel)] = (time, idx)

        if features == 'local':
            feature.append([mean_t / time_radius, mean_channel / channel_radius])
        elif features == 'global':
            feature.append([mean_t / time_window, mean_channel / num_channels])
    
    edges = torch.tensor(edges).t().contiguous()

    if features:
        feature = torch.tensor(feature)
        return edges, feature
    
    return edges



data = torch.load("datasets/hdspikes/processed/test/0.pt", weights_only=False)

num_channels = 700
channel_radius = 100.0
time_radius = 20000
time_window = 1000000
skip_channels = 10
features = "global"  # or "global"


data.pos[:, 0] = data.pos[:, 0] - data.pos[0, 0] # Start time from 0
data.pos[:, 0] *= 1e6 # Convert to microseconds
data.pos[:, 0] = torch.round(data.pos[:, 0]) # Round to nearest microsecond
data.pos = data.pos[data.pos[:, 0] < time_window] # Cut data to time window

from time import time


start = time()
edge_index1, feature1 = generate_edges(data.pos[:, 0], 
                                        data.pos[:, 1],
                                        features=features,
                                        time_radius=time_radius,
                                        time_window=time_window,
                                        channel_radius=channel_radius,
                                        skip_channels=skip_channels,
                                        num_channels=num_channels)

print("Time:", time() - start)

start = time()

edge_gen = edge_generator.EdgeGenerator(
    num_channels, channel_radius, time_radius, time_window, skip_channels, features
)
edge_index2, feature2 = edge_gen.generate_edges(data.pos[:, 0], data.pos[:, 1])

print("Time:", time() - start)
# print(feature2)
print("Max diff:", torch.max(torch.abs(feature1 - feature2)))
print("Edges equal:", torch.equal(edge_index1, edge_index2))
