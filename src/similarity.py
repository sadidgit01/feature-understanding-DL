import numpy as np


def get_query_indices(labels, classes):
    query_indices = []

    for class_name in classes:
        matches = np.where(labels == class_name)[0]
        if len(matches) == 0:
            continue
        query_indices.append(int(matches[0]))

    return query_indices


def find_top_k_neighbors(query_embedding, all_embeddings, k=10):
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    all_embeddings = np.asarray(all_embeddings, dtype=np.float32)

    query_norm = np.linalg.norm(query_embedding)
    all_norms = np.linalg.norm(all_embeddings, axis=1)

    similarities = all_embeddings @ query_embedding
    similarities = similarities / np.clip(all_norms * query_norm, a_min=1e-12, a_max=None)

    query_index = int(np.argmax(similarities))
    similarities[query_index] = -np.inf

    top_indices = np.argsort(similarities)[-k:][::-1]
    return top_indices


def run_similarity_search(embeddings, labels, paths, classes, k=10):
    results = []
    query_indices = get_query_indices(labels, classes)

    for query_index in query_indices:
        neighbor_indices = find_top_k_neighbors(embeddings[query_index], embeddings, k=k)
        results.append(
            {
                "query_path": paths[query_index],
                "query_label": labels[query_index],
                "neighbor_paths": paths[neighbor_indices].tolist(),
                "neighbor_labels": labels[neighbor_indices].tolist(),
            }
        )

    return results
