import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from celeb_dataset import CelebA_Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from segmentation_mask_overlay import overlay_masks
import os


def mask_prediction_visualization(batch, loader, model, folder='CelebA_mask_predictions/', device = 'cuda'):
    # this function visualizes the predicted masks

    if not os.path.exists(folder):
        os.makedirs(folder)
    model.eval()
    x, y = loader

    x = x.to(device)
    model.to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(x.unsqueeze(0))).squeeze(0)
        preds = (preds > 0.5).float()


        image = x.permute(1, 2, 0).cpu().numpy() # Permute dimensions and convert to numpy
        # [Example] Mimic list of masks
        masks = preds.cpu().numpy() # Convert masks to numpy
        # [Optional] prepare labels
        mask_labels = ['Skin', 'Nose', 'Hair']
        # [Optional] prepare colors
        cmap = plt.cm.tab20(np.arange(len(mask_labels)))[..., :-1]
        # Laminate your image!
        fig = overlay_masks(image, np.stack(masks, -1), mask_labels, return_type="mpl")
        fig.savefig(f"{folder}{batch}.png", bbox_inches="tight", dpi=300)

    model.train()