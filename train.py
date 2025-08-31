import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)


LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# BATCH_SIZE = 32
BATCH_SIZE = 4
NUM_EPOCHS = 3
# NUM_WORKERS = 2
NUM_WORKERS = 4
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images"
TRAIN_MASK_DIR = "data/train_masks"
VAL_IMG_DIR = "data/val_images"
VAL_MASK_DIR = "data/val_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        # data = data.to(DEVICE)
        data = data.to(DEVICE).half()
        # targets = targets.float().unsqueeze(1).to(device = DEVICE)
        targets = targets.half().unsqueeze(1).to(device = DEVICE)

        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
            predictions = model(data)

            loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )
    

    val_transform = A.Compose(
        [
            A.Resize(height = IMAGE_HEIGHT, width = IMAGE_WIDTH),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)

    check_accuracy(val_loader, model, device=DEVICE)
    


    scaler = torch.amp.GradScaler(device=DEVICE)

    print('Training...')
    print('Device:', DEVICE)
    for epoch in range(NUM_EPOCHS):

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # save the model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint,)

        save_predictions_as_imgs(val_loader, model, folder='saved_images/', device=DEVICE)
        



if __name__ == '__main__':
    main()