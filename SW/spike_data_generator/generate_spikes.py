
from pathlib import Path
import subprocess
import sys
import os
import re

# --------------------------------------
# Project paths
# --------------------------------------

GENERATOR_DIR = Path(__file__).resolve().parent
SW_DIR = GENERATOR_DIR.parent

AUDIO_DIR = SW_DIR / "audio"
OUTPUT_DIR = GENERATOR_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------
# German digit names
# --------------------------------------

GERMAN_DIGITS = {
    0: "null",
    1: "eins",
    2: "zwei",
    3: "drei",
    4: "vier",
    5: "fuenf",
    6: "sechs",
    7: "sieben",
    8: "acht",
    9: "neun",
}

# --------------------------------------
# Extract metadata from filename
# --------------------------------------

def extract_metadata(filename):
    stem = Path(filename).stem

    pattern = (
        r"^lang-(?P<language>[a-zA-Z]+)"
        r"_speaker-(?P<speaker>\d+)"
        r"_trial-(?P<trial>\d+)"
        r"_digit-(?P<digit>\d+)$"
    )

    match = re.fullmatch(pattern, stem)

    if not match:
        raise ValueError(
            "Invalid filename format: " + filename
        )

    data = match.groupdict()

    digit = int(data["digit"])
    language = data["language"].lower()

    if digit not in range(10):
        raise ValueError("Digit must be 0 to 9")

    data["digit"] = digit

    if language == "german":
        data["spoken_word"] = GERMAN_DIGITS[digit]
    else:
        data["spoken_word"] = "Not mapped"

    return data


# --------------------------------------
# Select FLAC file
# --------------------------------------

def select_audio():
    files = sorted(AUDIO_DIR.glob("*.flac"))

    if not files:
        raise FileNotFoundError(
            f"No FLAC files found in {AUDIO_DIR}"
        )

    print("\nAvailable FLAC files:\n")

    for index, file in enumerate(files, start=1):
        print(f"{index}. {file.name}")

    print("\nEnter file number or full filename.")

    choice = input("Select FLAC: ").strip()

    if choice.isdigit():
        index = int(choice)

        if not 1 <= index <= len(files):
            raise ValueError("Invalid selection number")

        return files[index - 1]

    # Only allow files inside the audio directory
    selected = AUDIO_DIR / Path(choice).name

    if selected not in files:
        raise FileNotFoundError(
            f"File not found: {selected}"
        )

    return selected


# --------------------------------------
# Generate spikes
# --------------------------------------

def generate_spikes(input_file):
    metadata = extract_metadata(input_file.name)

    output_file = (
        OUTPUT_DIR / f"{input_file.stem}_spikes.npz"
    )

    print("\n--------------------------------")
    print("Audio Metadata")
    print("--------------------------------")
    print("Language    :", metadata["language"])
    print("Speaker     :", metadata["speaker"])
    print("Trial       :", metadata["trial"])
    print("Digit       :", metadata["digit"])
    print("Spoken word :", metadata["spoken_word"])
    print("--------------------------------")

    env = os.environ.copy()

    env["PYTHONPATH"] = (
        str(GENERATOR_DIR)
        + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    command = [
        sys.executable,
        "-m",
        "lauscher",
        str(input_file),
        str(output_file),
        "--num_channels",
        "700",
        "--verbose",
    ]

    print("\nGenerating spikes...\n")

    subprocess.run(command, check=True, env=env)

    print("\nSpike generation completed!")
    print("Output:", output_file)


# --------------------------------------
# Main
# --------------------------------------

if __name__ == "__main__":
    selected_file = select_audio()
    generate_spikes(selected_file)
