from torch.utils.data import DataLoader

import dataset_loader
import preprocess
from config import BATCH_SIZE


def main():
    image_label_pairs = dataset_loader.load_image_paths()
    dataset = preprocess.ImageDataset(image_label_pairs, transform=preprocess.get_transform())
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    images, labels, paths = next(iter(dataloader))
    print(f"First batch shape: {images.shape}")
    print(f"First batch labels: {list(labels)}")
    print(f"First batch paths: {list(paths)}")


if __name__ == "__main__":
    main()
