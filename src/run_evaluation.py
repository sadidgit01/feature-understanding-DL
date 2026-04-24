import pickle
from pathlib import Path

from config import RESULTS_PATH
from evaluation import compute_all_metrics


MODEL_NAMES = ["resnet101", "googlenet", "zfnet"]


def print_comparison_table(all_metrics):
    header = f"{'Model':<12} {'NN Avg':>10} {'Intra Dist':>14} {'Inter Dist':>14}"
    separator = "-" * len(header)

    print("\nComparison Table")
    print(header)
    print(separator)
    for metrics in all_metrics:
        print(
            f"{metrics['model']:<12} "
            f"{metrics['nn_consistency_avg']:>10.4f} "
            f"{metrics['intra_class_distance']:>14.4f} "
            f"{metrics['inter_class_distance']:>14.4f}"
        )


def main():
    all_metrics = []

    for model_name in MODEL_NAMES:
        metrics = compute_all_metrics(model_name)
        all_metrics.append(metrics)

    print_comparison_table(all_metrics)

    output_path = Path(RESULTS_PATH) / "all_metrics.pkl"
    with output_path.open("wb") as file:
        pickle.dump(all_metrics, file)

    print(f"\nSaved metrics to {output_path}")


if __name__ == "__main__":
    main()
