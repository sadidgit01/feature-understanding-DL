import pickle
from pathlib import Path

import numpy as np

from config import EMBEDDINGS_PATH, RESULTS_PATH


def compute_nn_consistency(similarity_results):
    per_class_scores = {}

    for result in similarity_results:
        query_label = result["query_label"]
        neighbor_labels = result["neighbor_labels"]
        match_fraction = sum(label == query_label for label in neighbor_labels) / len(neighbor_labels)
        per_class_scores[query_label] = match_fraction

    overall_average = float(np.mean(list(per_class_scores.values()))) if per_class_scores else 0.0
    return per_class_scores, overall_average


def compute_intra_inter_distances(embeddings, labels):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)

    normalized = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    cosine_similarity = normalized @ normalized.T
    cosine_distance = 1.0 - cosine_similarity

    upper_triangle = np.triu(np.ones((len(labels), len(labels)), dtype=bool), k=1)
    same_class = labels[:, None] == labels[None, :]
    intra_mask = upper_triangle & same_class
    inter_mask = upper_triangle & (~same_class)

    intra_class_distance = float(cosine_distance[intra_mask].mean()) if np.any(intra_mask) else 0.0
    inter_class_distance = float(cosine_distance[inter_mask].mean()) if np.any(inter_mask) else 0.0

    return intra_class_distance, inter_class_distance


def compute_all_metrics(model_name):
    results_dir = Path(RESULTS_PATH)
    embeddings_dir = Path(EMBEDDINGS_PATH)

    with (results_dir / f"{model_name}_similarity_results.pkl").open("rb") as file:
        similarity_results = pickle.load(file)

    embeddings = np.load(embeddings_dir / f"{model_name}_embeddings.npy")
    labels = np.load(embeddings_dir / f"{model_name}_labels.npy", allow_pickle=True)

    nn_consistency_per_class, nn_consistency_avg = compute_nn_consistency(similarity_results)
    intra_class_distance, inter_class_distance = compute_intra_inter_distances(embeddings, labels)

    summary = {
        "model": model_name,
        "nn_consistency_per_class": nn_consistency_per_class,
        "nn_consistency_avg": nn_consistency_avg,
        "intra_class_distance": intra_class_distance,
        "inter_class_distance": inter_class_distance,
    }

    print(f"Model: {model_name}")
    print(f"  NN consistency per class: {nn_consistency_per_class}")
    print(f"  NN consistency average: {nn_consistency_avg:.4f}")
    print(f"  Intra-class cosine distance: {intra_class_distance:.4f}")
    print(f"  Inter-class cosine distance: {inter_class_distance:.4f}")

    return summary
