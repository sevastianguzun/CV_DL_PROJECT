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
    check_accuracy_per_class,
    save_predictions_as_imgs,
)
from visualizaton import mask_prediction_visualization
import torch.nn.functional as F


LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4
NUM_EPOCHS = 10
NUM_WORKERS = 4
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "CelebAMask-HQ/Celeb_dataset_train"
TRAIN_MASK_DIR = "CelebAMask-HQ/Celeb_dataset_masks_train"
TEST_IMG_DIR = "CelebAMask-HQ/Celeb_dataset_test"
TEST_MASK_DIR = "CelebAMask-HQ/Celeb_dataset_masks_test"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for params in model.parameters():
        params.requires_grad = False
    for params in model.final_conv.parameters():
        params.requires_grad = True
    for params in model.ups.parameters():
        params.requires_grad = True


    model.train()
    for batch_idx, (data, targets) in enumerate(loop):
        # data = data.to(DEVICE)
        data = data.to(DEVICE).half()
        # targets = targets.float().unsqueeze(1).to(device = DEVICE)
        targets = targets.half().to(device = DEVICE)

        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16):
            predictions = model(data)

            loss_bce = loss_fn[0](predictions, targets)
            loss_dice = loss_fn[1](predictions, targets)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss.item())



class Dice_loss:
    def __init__(self, smooth=1):
        self.smooth = smooth
    """
    Computes Dice Loss for multi-class segmentation.
    Args:
        pred: Tensor of predictions (batch_size, C, H, W).
        target: One-hot encoded ground truth (batch_size, C, H, W).
        smooth: Smoothing factor.
    Returns:
        Scalar Dice Loss.
    """
    def __call__(self, pred, target):
        pred = F.softmax(pred, dim=1)  # Convert logits to probabilities
        num_classes = pred.shape[1]  # Number of classes (C)
        dice = 0  # Initialize Dice loss accumulator

        for c in range(num_classes):  # Loop through each class
            pred_c = pred[:, c]  # Predictions for class c
            target_c = target[:, c]  # Ground truth for class c

            intersection = (pred_c * target_c).sum(dim=(1, 2))  # Element-wise multiplication
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))  # Sum of all pixels

            dice += (2. * intersection + self.smooth) / (union + self.smooth)  # Per-class Dice score

            return 1 - dice.mean() / num_classes

def main():
    torch.manual_seed(42)

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
        is_check_shapes=False,
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
        is_check_shapes=False,  
    )

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)

    if LOAD_MODEL:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'), model)
    

    model.final_conv = nn.Conv2d(64, 3, kernel_size=1).to(DEVICE)

    dice_loss = Dice_loss()
    bce_loss = nn.BCEWithLogitsLoss()
    loss_fn = [bce_loss, dice_loss]
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
        is_celeb_dataset= True,
    )

    print('Checking metrics before training...')
    check_accuracy_per_class(val_loader, model, device=DEVICE)

    x_test, y_test = next(iter(val_loader))

    for batch in range(BATCH_SIZE):
        mask_prediction_visualization(batch, loader=[x_test[batch], y_test[batch]],model=model, folder='CelebA_mask_predictions/before_finetuning/', device=DEVICE)

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

        if epoch + 1== 10:
            print('Saving checkpoint...')
            file_name = 'my_checkpoint_epoch_' + str(epoch + 1) + '.pth.tar'
            save_checkpoint(checkpoint, filename=file_name)

        print('Checking metrics for epoch', epoch + 1)
        check_accuracy_per_class(val_loader, model, device=DEVICE)
        
    print('Training complete!')

    print('Metrics after fine tuning:')
    check_accuracy_per_class(val_loader, model, device=DEVICE)

    for batch in range(BATCH_SIZE):
        mask_prediction_visualization(batch, loader=[x_test[batch], y_test[batch]],model=model, folder='CelebA_mask_predictions/after_finetuning/', device=DEVICE)


if __name__ == '__main__':
    main()