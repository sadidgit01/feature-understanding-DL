import torch

from config import DEVICE
from models import load_all_models


def main():
    models = load_all_models()
    sample_shape = (1, 3, 224, 224)
    sample = torch.randn(*sample_shape, device=DEVICE)

    for model_name, model in models.items():
        with torch.no_grad():
            output = model(sample)
        print(f"{model_name}: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
