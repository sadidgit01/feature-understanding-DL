from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset_loader
import models
import preprocess
from config import BATCH_SIZE, DEVICE, EMBEDDINGS_PATH


def extract_model_embeddings(model_name, model, dataloader, output_dir):
    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc=f"Extracting {model_name}", unit="batch"):
            images = images.to(DEVICE)
            outputs = model(images)

            all_embeddings.append(outputs.cpu().numpy())
            all_labels.extend(labels)
            all_paths.extend(paths)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)
    paths = np.array(all_paths)

    np.save(output_dir / f"{model_name}_embeddings.npy", embeddings)
    np.save(output_dir / f"{model_name}_labels.npy", labels)
    np.save(output_dir / f"{model_name}_paths.npy", paths)

    print(f"Saved {model_name} embeddings with shape {embeddings.shape}")


def main():
    output_dir = Path(EMBEDDINGS_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_label_pairs = dataset_loader.load_image_paths()
    dataset = preprocess.ImageDataset(image_label_pairs, transform=preprocess.get_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_models = models.load_all_models()

    for model_name, model in all_models.items():
        extract_model_embeddings(model_name, model, dataloader, output_dir)


if __name__ == "__main__":
    main()
