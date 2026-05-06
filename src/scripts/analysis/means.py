import argparse
import json
import numpy as np


def calculate_means(filepath: str) -> None:
    with open(filepath, "r") as f:
        data = json.load(f)

    model_name = data.get("model", "Unknown model")
    print(f"Model: {model_name}\n")

    skip_keys = {"model"}
    for key, value in data.items():
        if key in skip_keys:
            continue
        if isinstance(value, list) and value:
            mean = sum(value) / len(value)
            std = np.std(value)
            print(f"{key}: {mean:.6f}\t{std:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate mean values of metric arrays from a JSON results file."
    )
    parser.add_argument("-i", type=str, help="Path to the JSON file")
    args = parser.parse_args()

    calculate_means(args.i)