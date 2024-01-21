import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from tqdm import tqdm
import clip

torch.set_grad_enabled(False)

DEVICE = torch.device("gpu") if torch.cuda.is_available() else torch.device("cpu")

# MODELS = {
#     "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
#     "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
#     "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
#     "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
#     "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
#     "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
#     "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
#     "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
#     "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
# }

MODEL_NAME = "ViT-L/14"
model, preprocess = clip.load(MODEL_NAME, device=DEVICE)

for dataset_name in ["CIFAR100", "ImageNet"]:
    for split in ["train", "val"]:
        if dataset_name == "ImageNet":
            from torchvision.datasets import ImageNet

            dataset = ImageNet(
                root="datasets\ImageNet",
                split=split,
                download=True,
                transform=preprocess,
            )
        else:
            from torchvision.datasets import CIFAR100

            dataset = CIFAR100(
                root="datasets\CIFAR100",
                train=split == "train",
                download=True,
                transform=preprocess,
            )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    embeddings = []
    labels = []
    for images, batch_labels in tqdm(dataloader, colour="magenta"):

        batch_embeddings = model.encode_image(images.to(DEVICE))

        embeddings.append(batch_embeddings.cpu().numpy())
        labels.append(batch_labels.numpy())

    np.save(
        f"embeddings/embeddings_clip-{MODEL_NAME}_{dataset_name}-{split}.npy",
        embeddings,
    )
    np.save(f"embeddings/labels_clip-{MODEL_NAME}_{dataset_name}-{split}.npy", labels)

print("End")
