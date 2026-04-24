import pickle
from pathlib import Path

import pandas as pd

from config import RESULTS_PATH


def main():
    results_dir = Path(RESULTS_PATH)
    metrics_path = results_dir / "all_metrics.pkl"

    with metrics_path.open("rb") as file:
        all_metrics = pickle.load(file)

    summary_rows = []
    for metrics in all_metrics:
        separation_gap = metrics["inter_class_distance"] - metrics["intra_class_distance"]
        summary_rows.append(
            {
                "Model": metrics["model"],
                "NN Consistency Avg": metrics["nn_consistency_avg"],
                "Intra-class Distance": metrics["intra_class_distance"],
                "Inter-class Distance": metrics["inter_class_distance"],
                "Separation Gap": separation_gap,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(by="NN Consistency Avg", ascending=False).reset_index(drop=True)

    print("Final Summary Table")
    print(summary_df.to_string(index=False))

    ranking = summary_df["Model"].tolist()
    if len(ranking) >= 3:
        print(f"\nFinal Verdict: 1st = {ranking[0]}, 2nd = {ranking[1]}, 3rd = {ranking[2]}")
    else:
        print(f"\nFinal Verdict: ranking unavailable for fewer than 3 models")

    output_path = results_dir / "final_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Saved summary CSV to {output_path}")


if __name__ == "__main__":
    main()
