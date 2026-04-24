import pickle
from pathlib import Path

import numpy as np

from config import CLASSES, EMBEDDINGS_PATH, RESULTS_PATH
from similarity import run_similarity_search


MODEL_NAMES = ["resnet101", "googlenet", "zfnet"]


def load_model_arrays(model_name):
    embeddings_dir = Path(EMBEDDINGS_PATH)
    embeddings = np.load(embeddings_dir / f"{model_name}_embeddings.npy")
    labels = np.load(embeddings_dir / f"{model_name}_labels.npy", allow_pickle=True)
    paths = np.load(embeddings_dir / f"{model_name}_paths.npy", allow_pickle=True)
    return embeddings, labels, paths


def main():
    results_dir = Path(RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_NAMES:
        embeddings, labels, paths = load_model_arrays(model_name)
        results = run_similarity_search(embeddings, labels, paths, CLASSES, k=10)

        print(f"Model: {model_name}")
        for result in results:
            same_class_count = sum(label == result["query_label"] for label in result["neighbor_labels"])
            print(f"  Query {result['query_label']}: {same_class_count}/10 neighbors match the same class")

        output_path = results_dir / f"{model_name}_similarity_results.pkl"
        with output_path.open("wb") as file:
            pickle.dump(results, file)


if __name__ == "__main__":
    main()
