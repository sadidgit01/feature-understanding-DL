import torch


DATASET_PATH = "dataset/raw"
EMBEDDINGS_PATH = "embeddings/"
PLOTS_PATH = "plots/"
RESULTS_PATH = "results/"
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea"]
NUM_IMAGES_PER_CLASS = 200
IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
