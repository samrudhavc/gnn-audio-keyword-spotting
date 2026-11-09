
from pathlib import Path
import subprocess
import os
import numpy as np

# Project directories
SW_DIR = Path(__file__).resolve().parent
AUDIO_DIR = SW_DIR / "audio"
GENERATOR_DIR = SW_DIR / "spike_data_generator"
OUTPUT_DIR = GENERATOR_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Python environment used for Lauscher
LAUSCHER_PYTHON = (
    Path.home()
    / "MAINPROJECT/RepoGNN/lauscher-master/.lausenv/bin/python"
)


def select_flac():
    files = sorted(AUDIO_DIR.glob("*.flac"))

    if not files:
        raise FileNotFoundError(
            f"No FLAC files found in {AUDIO_DIR}"
        )

    print("\nAvailable FLAC files:\n")

    for i, file in enumerate(files, 1):
        print(f"{i}. {file.name}")

    choice = input("\nEnter number or filename: ").strip()

    if choice.isdigit():
        index = int(choice) - 1

        if index < 0 or index >= len(files):
            raise ValueError("Invalid file number")

        return files[index]

    selected = AUDIO_DIR / Path(choice).name

    if selected not in files:
        raise FileNotFoundError(
            f"Audio file not found: {selected}"
        )

    return selected


def generate_spikes(flac_file):
    npz_file = OUTPUT_DIR / f"{flac_file.stem}_spikes.npz"

    # Reuse existing spikes
    if npz_file.exists():
        print("\nUsing existing spike file:")
        print(npz_file)
        return npz_file

    print("\nGenerating spikes from:", flac_file.name)

    if not LAUSCHER_PYTHON.is_file():
        raise FileNotFoundError(
            f"Lauscher Python not found: {LAUSCHER_PYTHON}"
        )

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(GENERATOR_DIR)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    command = [
        str(LAUSCHER_PYTHON),
        "-m",
        "lauscher",
        str(flac_file),
        str(npz_file),
        "--num_channels",
        "700",
        "--verbose",
    ]

    subprocess.run(command, check=True, env=env)

    return npz_file


def inspect_spikes(npz_file):
    with np.load(npz_file, allow_pickle=False) as data:
        spikes = data["arr_0"]

    if spikes.ndim != 2 or spikes.shape[0] != 2:
        raise ValueError(
            f"Unexpected spike array shape: {spikes.shape}"
        )

    timestamps = spikes[0]
    channels = spikes[1].astype(np.int64)

    print("\n========== SPIKE DATA ==========")
    print("File:", npz_file.name)
    print("Total events:", len(timestamps))
    print("First timestamps:", timestamps[:10])
    print("First channels:", channels[:10])
    print("================================")

    return timestamps, channels


def main():
    flac_file = select_flac()

    print("\nSelected audio:", flac_file.name)

    npz_file = generate_spikes(flac_file)

    timestamps, channels = inspect_spikes(npz_file)

    print("\nFLAC to spike conversion completed.")
    print("Saved spike file:", npz_file)


if __name__ == "__main__":
    main()
