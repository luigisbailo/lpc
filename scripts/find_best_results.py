#!/usr/bin/env python3
import argparse
import os
import glob
import pickle
import numpy as np


def parse_args():
    """Parse command‐line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate best model results from multiple subdirectories."
    )
    parser.add_argument(
        "--results-dir", required=True, help="Directory containing model results."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save aggregated results."
    )
    return parser.parse_args()


def get_best_result_from_dir(model, subdir_path):
    """
    For a given model and a subdirectory, find the pickle file with the highest test accuracy.
    
    The function searches for files matching the pattern <model>*.pkl. For each file, it
    constructs a corresponding checkpoint file name, loads the result dictionary, and then
    selects the one with the highest accuracy (assumed to be stored in res['accuracy_test'][-1][0]).
    
    Parameters:
        model (str): The model name (e.g., 'lpc', 'wide_lpc', etc.)
        subdir_path (str): Path to the subdirectory to search within.
    
    Returns:
        dict or None: The result dictionary corresponding to the highest accuracy in the
                      subdirectory, or None if no matching file was found.
    """
    best_result = None
    max_accuracy = 0
    file_pattern = os.path.join(subdir_path, f"{model}*.pkl")
    for file_path in glob.glob(file_pattern):
        # Construct the corresponding checkpoint filename
        checkpoint_file = (
            file_path.replace(model, f"checkpoint_{model}").replace(".pkl", ".pth.tar")
        )
        with open(file_path, "rb") as f:
            res = pickle.load(f)
        res["checkpoint_file"] = checkpoint_file

        # Get accuracy from the result dictionary
        accuracy = res["accuracy_test"][-1][0]
        if accuracy > max_accuracy:
            best_result = res
            max_accuracy = accuracy

    return best_result


def aggregate_best_results(best_results):
    """
    Aggregate a list of best result dictionaries by concatenating corresponding values.
    
    For each key in the result dictionaries:
      - If the value is a dict, each subkey’s arrays are horizontally stacked.
      - If the key is 'entropy_train' or 'entropy_test', a special squeeze-and-select operation is performed.
      - The keys 'penultimate_train' and 'penultimate_test' are skipped.
      - Otherwise, the arrays are concatenated using np.hstack.
    
    Parameters:
        best_results (list of dict): A list of result dictionaries.
    
    Returns:
        dict: A single dictionary containing aggregated values.
    """
    aggregated = {}
    keys_to_skip = {"penultimate_train", "penultimate_test"}

    # Loop over the keys from the first result dictionary
    for key, value in best_results[0].items():
        if key in keys_to_skip:
            continue
        if isinstance(value, dict):
            # For nested dictionaries, stack each subkey's data horizontally.
            aggregated[key] = {
                subkey: np.hstack([res[key][subkey] for res in best_results])
                for subkey in value.keys()
            }
        else:
            aggregated[key] = np.hstack([res[key] for res in best_results])
    return aggregated


def process_model_results(model, results_dir):
    """
    Process results for a given model across all subdirectories in results_dir.
    
    This function iterates over each subdirectory (ignoring '.ipynb_checkpoints'), obtains the best
    result from each, and then aggregates these results.
    
    Parameters:
        model (str): The model name.
        results_dir (str): The directory containing subdirectories of results.
    
    Returns:
        dict or None: The aggregated result dictionary for the model, or None if no results are found.
    """
    best_results = []
    for entry in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, entry)
        if not os.path.isdir(subdir_path) or entry == ".ipynb_checkpoints":
            continue
        print(f"Processing subdirectory '{entry}' for model '{model}'...")
        best_res = get_best_result_from_dir(model, subdir_path)
        if best_res is not None:
            best_results.append(best_res)

    if not best_results:
        print(f"No results found for model '{model}'.")
        return None

    return aggregate_best_results(best_results)


def main():
    args = parse_args()
    results_dir = args.results_dir
    output_dir = args.output_dir

    models = [
        "lpc_lin_pen",
        "wide_lpc_lin_pen",
        "narrow_lpc_lin_pen",
        "scl_lpc_lin_pen",
        "lpc_no_pen",
        "no_pen",
        "scl_no_pen",
        "arcface_no_pen",
        "lin_pen",
        "nonlin_pen",
    ]

    aggregated_results = {}
    for model in models:
        print(f"\n=== Processing model: {model} ===")
        aggregated = process_model_results(model, results_dir)
        if aggregated is not None:
            aggregated_results[model] = aggregated

    output_file = os.path.join(output_dir, "best_results.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(aggregated_results, f)
    print(f"\nAggregated results saved to {output_file}")


if __name__ == "__main__":
    main()
