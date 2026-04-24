import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

import dataset_loader
import extract_embeddings
import generate_report_data
import models
import preprocess
import visualize
from config import BATCH_SIZE, CLASSES, EMBEDDINGS_PATH, RESULTS_PATH
from evaluation import compute_all_metrics
from similarity import run_similarity_search


MODEL_NAMES = ["resnet101", "googlenet", "zfnet"]


def run_similarity_stage():
    results_dir = Path(RESULTS_PATH)
    results_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_NAMES:
        embeddings = np.load(Path(EMBEDDINGS_PATH) / f"{model_name}_embeddings.npy")
        labels = np.load(Path(EMBEDDINGS_PATH) / f"{model_name}_labels.npy", allow_pickle=True)
        paths = np.load(Path(EMBEDDINGS_PATH) / f"{model_name}_paths.npy", allow_pickle=True)

        results = run_similarity_search(embeddings, labels, paths, CLASSES, k=10)
        with (results_dir / f"{model_name}_similarity_results.pkl").open("wb") as file:
            pickle.dump(results, file)


def run_evaluation_stage():
    all_metrics = []

    for model_name in MODEL_NAMES:
        metrics = compute_all_metrics(model_name)
        all_metrics.append(metrics)

    with (Path(RESULTS_PATH) / "all_metrics.pkl").open("wb") as file:
        pickle.dump(all_metrics, file)


def main():
    print("Step 1: Loading dataset...")
    image_label_pairs = dataset_loader.load_image_paths()

    print("Step 2: Loading models...")
    loaded_models = models.load_all_models()

    print("Step 3: Extracting embeddings...")
    dataset = preprocess.ImageDataset(image_label_pairs, transform=preprocess.get_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    output_dir = Path(EMBEDDINGS_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in loaded_models.items():
        extract_embeddings.extract_model_embeddings(model_name, model, dataloader, output_dir)

    print("Step 4: Running similarity search...")
    run_similarity_stage()

    print("Step 5: Computing evaluation metrics...")
    run_evaluation_stage()

    print("Step 6: Generating visualizations...")
    visualize.generate_visualizations()

    print("Step 7: Generating final report data...")
    generate_report_data.main()

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
