#!/usr/bin/env python3
"""Select a FLAC recording, reuse/generate Lauscher spikes, and classify with SHD GNN.
Run from SW with the GNN Python environment active.
"""
from pathlib import Path
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import torch
import yaml
from dotmap import DotMap
from torch_geometric.data import Batch, Data

SW = Path(__file__).resolve().parent
AUDIO = SW / 'audio'
GEN = SW / 'spike_data_generator'
OUT = GEN / 'outputs'
DEFAULT_CKPT = SW / 'checkpoints' / 'best_model_float-v1.ckpt'
# DEFAULT_LAUSCHER_PYTHON = SW.parent.parent / 'lauscher-master' / '.lausenv' / 'bin' / 'python'
DEFAULT_LAUSCHER_PYTHON = Path(sys.executable)

def choose_file():
    files = sorted(AUDIO.glob('*.flac'))
    if not files:
        raise FileNotFoundError(f'No .flac files in {AUDIO}')
    print('\nAvailable FLAC recordings:')
    for i, file in enumerate(files, 1):
        print(f'{i:3}. {file.name}')
    choice = input('\nEnter number or filename: ').strip()
    if choice.isdigit() and 1 <= int(choice) <= len(files):
        return files[int(choice) - 1]
    path = AUDIO / Path(choice).name
    if not path.is_file():
        raise FileNotFoundError(f'FLAC not found: {path}')
    return path


def metadata(path):
    match = re.search(r'lang-([a-zA-Z]+).*?_digit-(\d+)(?:$|_)', path.stem)
    if match:
        return match.group(1).lower(), int(match.group(2))
    return None, None


def spikes_for(flac, lauscher_python, regenerate=False):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f'{flac.stem}_spikes.npz'
    if target.is_file() and not regenerate:
        print(f'Using existing spikes: {target}')
        return target
    if not lauscher_python.is_file():
        raise FileNotFoundError(f'Lauscher Python not found: {lauscher_python}')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(GEN) + os.pathsep + env.get('PYTHONPATH', '')
    print(f'Generating spikes from {flac.name} ...', flush=True)
    subprocess.run([str(lauscher_python), '-m', 'lauscher', str(flac), str(target),
                    '--num_channels', '700', '--verbose'], check=True, env=env)
    return target


def graph_from_npz(path, cfg):
    import edge_generator
    with np.load(path, allow_pickle=False) as z:
        if 'arr_0' not in z:
            raise ValueError(f'Expected arr_0 in NPZ; found {z.files}')
        spikes = z['arr_0']
    if spikes.ndim != 2 or spikes.shape[0] != 2:
        raise ValueError(f'Expected spike array (2, N), got {spikes.shape}')
    if not np.isfinite(spikes).all():
        raise ValueError('Spike array contains NaN or infinite values')
    if spikes.shape[1] == 0:
        raise ValueError('No spike events in file')
    seconds = spikes[0]
    channels = spikes[1]
    if not np.allclose(channels, np.rint(channels), atol=1e-6):
        raise ValueError('Channel IDs are not integers')
    if np.any(channels < 0) or np.any(channels >= cfg.general.num_channels):
        raise ValueError('Channel IDs outside configured range')
    order = np.argsort(seconds, kind='stable')
    seconds, channels = seconds[order], channels[order]
    # Match SpikingDS.__getitem__: shift to first event, convert to µs, round, crop.
    times = np.rint((seconds - seconds[0]) * 1e6)
    keep = times < cfg.general.time_window
    times = times[keep]
    channels = channels[keep]
    if len(times) == 0:
        raise ValueError('No events remain after time-window cropping')
    time_tensor = torch.tensor(times, dtype=torch.float32)
    channel_tensor = torch.tensor(channels, dtype=torch.float32)
    edge_gen = edge_generator.EdgeGenerator(
        cfg.general.num_channels,
        cfg.graph.channel_radius,
        cfg.graph.time_radius,
        cfg.general.time_window,
        cfg.graph.skip_channels,
        cfg.graph.features,
    )
    edges, features = edge_gen.generate_edges(time_tensor, channel_tensor)
    pos = torch.stack((time_tensor / cfg.general.time_window,
                       channel_tensor / cfg.general.num_channels), dim=1)
    data = Data(pos=pos, edge_index=edges.long())
    if features is not None:
        data.x = features.float()
    return data


def main():
    parser = argparse.ArgumentParser(description='Predict spoken digit using Lauscher + trained SHD GNN')
    parser.add_argument('--file', help='FLAC filename in SW/audio; otherwise interactive selection')
    parser.add_argument('--npz', type=Path, help='Use an existing NPZ directly (skip FLAC conversion)')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CKPT)
    parser.add_argument('--lauscher-python', type=Path, default=DEFAULT_LAUSCHER_PYTHON)
    parser.add_argument('--regenerate', action='store_true', help='Regenerate spikes even if NPZ exists')
    args = parser.parse_args()

    flac = None
    if args.npz:
        npz = args.npz.expanduser().resolve()
        if not npz.is_file():
            raise FileNotFoundError(npz)
        language, expected = metadata(npz)
    else:
        flac = AUDIO / Path(args.file).name if args.file else choose_file()
        if not flac.is_file():
            raise FileNotFoundError(flac)
        language, expected = metadata(flac)
        npz = spikes_for(flac, args.lauscher_python.expanduser().resolve(), args.regenerate)

    with (SW / 'configs' / 'digits.yaml').open() as stream:
        cfg = DotMap(yaml.safe_load(stream))
    data = graph_from_npz(npz, cfg)
    from models.recognition import LNRecognition
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    print(f'Loading trusted checkpoint: {checkpoint}')
    model = LNRecognition.load_from_checkpoint(str(checkpoint), config=cfg, weights_only=False,
                                               map_location='cpu')
    model.eval()
    batch = Batch.from_data_list([data])
    with torch.inference_mode():
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)
        class_id = int(probabilities.argmax().item())
        confidence = float(probabilities[class_id].item())

    print('\n===== GNN RESULT =====')
    print('Spike file:', npz.name)
    print('Graph nodes:', data.num_nodes)
    print('Graph edges:', data.edge_index.shape[1])
    print('Predicted SHD class ID:', class_id)
    print(f'Model softmax score: {confidence:.2%}')
    print('Filename language:', language if language else 'unknown')
    print('Filename digit:', expected if expected is not None else 'unknown')
    print('NOTE: Class ID is not yet mapped to language/digit. Do not compare directly')
    print('until SHD class ordering is verified from the dataset labels.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
