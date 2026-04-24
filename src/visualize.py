from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from config import CLASSES, EMBEDDINGS_PATH, PLOTS_PATH


matplotlib.use("Agg")


MODEL_NAMES = ["resnet101", "googlenet", "zfnet"]


def load_embeddings_and_labels(model_name):
    embeddings_dir = Path(EMBEDDINGS_PATH)
    embeddings = np.load(embeddings_dir / f"{model_name}_embeddings.npy")
    labels = np.load(embeddings_dir / f"{model_name}_labels.npy", allow_pickle=True)
    return embeddings, labels


def reduce_with_tsne(embeddings):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    return tsne.fit_transform(embeddings)


def plot_tsne(ax, reduced_embeddings, labels, model_name):
    for class_name in CLASSES:
        class_mask = labels == class_name
        ax.scatter(
            reduced_embeddings[class_mask, 0],
            reduced_embeddings[class_mask, 1],
            label=class_name,
            s=18,
            alpha=0.8,
        )

    ax.set_title(model_name)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()


def generate_visualizations():
    plots_dir = Path(PLOTS_PATH)
    plots_dir.mkdir(parents=True, exist_ok=True)

    reduced_results = {}

    for model_name in MODEL_NAMES:
        embeddings, labels = load_embeddings_and_labels(model_name)
        reduced_embeddings = reduce_with_tsne(embeddings)
        reduced_results[model_name] = (reduced_embeddings, labels)

        figure, axis = plt.subplots(figsize=(8, 6))
        plot_tsne(axis, reduced_embeddings, labels, model_name)
        output_path = plots_dir / f"{model_name}_tsne.png"
        figure.tight_layout()
        figure.savefig(output_path, dpi=300)
        plt.close(figure)
        print(f"Saved plot: {output_path}")

    combined_figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    for axis, model_name in zip(axes, MODEL_NAMES):
        reduced_embeddings, labels = reduced_results[model_name]
        plot_tsne(axis, reduced_embeddings, labels, model_name)

    combined_figure.tight_layout()
    comparison_path = plots_dir / "tsne_comparison.png"
    combined_figure.savefig(comparison_path, dpi=300)
    plt.close(combined_figure)
    print(f"Saved plot: {comparison_path}")


if __name__ == "__main__":
    generate_visualizations()
