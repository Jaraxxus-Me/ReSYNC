"""Script to modify Failure Dataset by slicing states from index 22 onwards.

Loads an existing Failure Dataset, modifies the states to only include
original_states[:, 22:], and saves a new version.
"""

import argparse
import logging
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modify Failure Dataset by slicing states"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="training_data/cluttered_room/Failure_data/scenario_1/seed_0/state_labels.pt",
        help="Path to input dataset file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save modified dataset. Defaults to input directory with '_sliced' suffix",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=22,
        help="Start index for state slicing (default: 22)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_path = Path(args.input_path)
    if args.output_path is None:
        output_path = input_path.parent / f"state_labels_sliced_{args.start_idx}.pt"
    else:
        output_path = Path(args.output_path)

    logging.info(f"Loading dataset from {input_path}")
    data = torch.load(input_path, map_location="cpu")

    states = data["states"]
    labels = data["labels"]

    logging.info(f"Original dataset: {len(states)} states")

    # Process states - slice from start_idx onwards
    modified_states = []
    for _, state in enumerate(states):
        if state.dim() == 1:
            # Shape: [state_dim] -> slice to [state_dim - start_idx]
            modified_state = state[args.start_idx :]
        else:
            # Shape: [batch, state_dim] or similar -> slice second dimension
            modified_state = state[..., args.start_idx :]
        modified_states.append(modified_state)

    # Log the shape change
    if len(states) > 0:
        orig_shape = states[0].shape
        new_shape = modified_states[0].shape
        logging.info(f"State shape changed: {orig_shape} -> {new_shape}")

    # Create new dataset dictionary
    modified_data = {"states": modified_states, "labels": labels}

    # Save modified dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modified_data, output_path)
    logging.info(f"Saved modified dataset to {output_path}")


if __name__ == "__main__":
    main()
