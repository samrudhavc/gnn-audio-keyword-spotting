# Spike Data Generator – Lauscher Artificial Cochlea

## Overview

The `spike_data_generator` module is part of the **Hardware-Accelerated Graph Neural Networks for Event-Based Audio Classification and Keyword Spotting on FPGA** project.

This module converts conventional digital audio recordings into spike-based auditory events using the **Lauscher artificial cochlea model**.

The generated spike events serve as an intermediate representation for graph construction and subsequent classification using a Graph Neural Network (GNN).

The implementation is based on the Lauscher audio-to-spike conversion software associated with the Heidelberg Spiking Datasets.

## Objectives

- Convert FLAC audio recordings into spike-based auditory events.
- Generate event representations compatible with the intended GNN preprocessing pipeline.
- Analyze spike timestamps and auditory channel information.
- Save generated events in NumPy `.npz` format.
- Prepare for integration with graph construction and GNN inference.
- Support future development of a real-time audio classification system on FPGA.

## Processing Architecture

```text
FLAC Audio Recording
        |
        v
Lauscher Artificial Cochlea
        |
        v
Auditory Spike Generation
        |
        v
Spike Data (.npz)
        |
        v
Graph Construction
        |
        v
Graph Neural Network
        |
        v
Audio / Spoken Digit Classification
```

The FLAC-to-spike conversion stage is implemented using Lauscher. Graph construction and GNN classification are separate stages of the complete project.

## Directory Structure

```text
SW/
├── audio/
│   └── digit2.flac
│
├── spike_data_generator/
│   ├── lauscher/
│   │   ├── __main__.py
│   │   └── ...
│   ├── generate_spikes.py
│   ├── outputs/
│   │   └── digit2_spikes.npz
│   └── README.md
│
├── configs/
├── data/
├── models/
├── checkpoints/
├── train.py
└── test.py
```

## Software Requirements

- Fedora Linux or another compatible Linux distribution
- Python 3.11
- NumPy
- Lauscher and its required Python dependencies
- FLAC audio recordings

The Lauscher software is maintained separately from the main GNN environment to avoid dependency conflicts.

## Installation

### 1. Navigate to the SW directory

```bash
cd ~/MAINPROJECT/RepoGNN/gcnn-audio-fpga-main/SW
```

### 2. Activate the Lauscher virtual environment

```bash
source ~/MAINPROJECT/RepoGNN/lauscher-master/.lausenv/bin/activate
```

### 3. Prepare the output directory

```bash
mkdir -p spike_data_generator/outputs
```

The `lauscher` Python package must be available inside `spike_data_generator/`, or otherwise installed in the active Python environment.

## Generating Spike Data

The `generate_spikes.py` wrapper invokes the Lauscher conversion program.

Run the following command from the `SW` directory:

```bash
PYTHONPATH="$PWD/spike_data_generator" \
python spike_data_generator/generate_spikes.py
```

The current wrapper uses:

**Input:**

```text
SW/audio/digit2.flac
```

**Output:**

```text
SW/spike_data_generator/outputs/digit2_spikes.npz
```

The configured number of auditory channels is 700, matching the channel count used in the SHD dataset.

This channel count alone does not guarantee exact equivalence with the original SHD spike encoding. Other cochlea and preprocessing parameters must also be verified.

## Output Verification

To check whether the output file was generated:

```bash
ls -lh spike_data_generator/outputs/
```

To inspect the NumPy arrays:

```python
import numpy as np

path = "spike_data_generator/outputs/digit2_spikes.npz"

with np.load(path, allow_pickle=False) as data:
    print("Available arrays:", data.files)

    for name in data.files:
        array = data[name]

        print("Name:", name)
        print("Shape:", array.shape)
        print("Data type:", array.dtype)
        print("First values:", array[:10] if array.ndim else array)
```

The actual array names and structures should be confirmed from the generated file before implementing the graph conversion module.

## Relationship with the GNN Model

The existing GNN model is trained using event-based audio data from the Spiking Heidelberg Digits dataset.

The trained model uses graph representations derived from auditory events.

The intended integration is:

1. Read the Lauscher-generated spike events.
2. Extract event timestamps and auditory channel identifiers.
3. Apply the graph preprocessing used by the existing GNN dataset pipeline.
4. Construct graph nodes and edges.
5. Load the trained GNN checkpoint.
6. Perform inference and obtain the predicted spoken-digit class.

The graph adapter must preserve the feature dimensions, timestamp units, normalization and neighborhood construction expected by the trained model.

## Current Implementation Status

**Implemented or previously tested:**

- Lauscher environment setup
- FLAC audio processing and spike visualization
- Initial audio-to-spike conversion experiments
- `generate_spikes.py` wrapper preparation
- Trained SHD GNN model available for subsequent testing

**Integration and validation pending:**

- Verification of the saved `.npz` event structure
- Confirmation of cochlea parameter compatibility with SHD
- Conversion of Lauscher events into GNN-compatible graphs
- End-to-end classification of external FLAC recordings
- Evaluation of classification accuracy on generated spike data

## Future Development

### Real-Time Audio Processing

The proposed future pipeline is:

```text
Microphone
    |
    v
Audio Acquisition
    |
    v
Artificial Cochlea / Event Encoder
    |
    v
Event Stream
    |
    v
Graph Generator
    |
    v
GNN Accelerator
    |
    v
Keyword Classification
```

The current Lauscher implementation is used for offline audio-to-spike conversion. Real-time performance has not yet been established.

Further work will investigate streaming audio acquisition, event-generation latency, graph processing and FPGA hardware acceleration.

### FPGA Integration

The target development platform is the Nexys Video Artix-7 FPGA.

Planned hardware development includes evaluating graph generation, quantized GNN inference, memory requirements and communication between audio acquisition and the FPGA processing stages.

## References and Attribution

1. Cramer, B., et al., *The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks*, IEEE Transactions on Neural Networks and Learning Systems, 2022.
2. Lauscher audio-to-spike conversion software 
<!-- : https://github.com/electronicvisions/lauscher -->
3. GCNN Audio FPGA reference implementation
<!-- : https://github.com/vision-agh/gcnn-audio-fpga -->

Lauscher is third-party software. Its original authorship, copyright notices and license requirements must be preserved when including its source code in this repository.

## Project Context

This module supports the M.Tech main project:

**Hardware-Accelerated Graph Neural Networks: An Alternative Approach for Event-Based Audio Classification and Keyword Spotting on SoC FPGA**

Its purpose is to connect conventional audio recordings with event-driven graph neural network processing, forming a foundation for future real-time embedded audio classification.