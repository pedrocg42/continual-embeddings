import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm

torch.set_grad_enabled(False)

DEVICE = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")

# MODELS = [
#     "dinov2_vits14",
#     'dinov2_vitb14',
#     "dinov2_vitl14",
#     'dinov2_vitg14',
#     "dinov2_vits14_reg",
#     'dinov2_vitb14_reg',
#     "dinov2_vitl14_reg",
#     'dinov2_vitg14_reg',
# ]
MODEL_NAME = 'dinov2_vitl14'
dinov2 = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)

dinov2.to(DEVICE)

for dataset_name in ["CIFAR100", "ImageNet"]:
    if dataset_name == "ImageNet":
        from torchvision.datasets import ImageNet

        dataset = ImageNet(
            root="datasets\ImageNet",
            split="val",
            download=True,
            transform=Compose([ToTensor(), Resize(224, antialias=True)]),
        )
    else:
        from torchvision.datasets import CIFAR100

        dataset = CIFAR100(
            root="datasets\CIFAR100",
            train=False,
            download=True,
            transform=Compose([ToTensor(), Resize(224, antialias=True)]),
        )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    embeddings = []
    labels = []
    for images, batch_labels in tqdm(dataloader, colour="magenta"):
        images.to(DEVICE)

        batch_embeddings = dinov2(images)

        embeddings.append(batch_embeddings.cpu().numpy())
        labels.append(batch_labels.numpy())

    np.save(f"embeddings_{MODEL_NAME}_{dataset_name}.npy", embeddings)
    np.save(f"labels_{MODEL_NAME}_{dataset_name}.npy", labels)

print("End")
