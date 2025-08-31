import torch
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from celeb_dataset import CelebA_Dataset


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('-> Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print('-> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
        is_celeb_dataset = False):
    
    if is_celeb_dataset:
        train_dataset = CelebA_Dataset(
            image_dir=train_dir,
            mask_dir=train_maskdir,
            transform=train_transform,
        )

        test_dataset = CelebA_Dataset(
            image_dir=val_dir,
            mask_dir=val_maskdir,
            transform=val_transform,
            is_test=True
        )

    else:

        train_dataset = CarvanaDataset(
            image_dir=train_dir,
            mask_dir=train_maskdir,
            transform=train_transform,
        )

        test_dataset = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir, 
        transform=val_transform,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                preds = torch.sigmoid(model(x))
            
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
    
    print(
        f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}%'
    )

    print(f'Dice score: {dice_score/len(loader):.2f}')

    model.train()

def save_predictions_as_imgs(loader, model, folder='saved_images/', device = 'cuda'):
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f'{folder}/pred_{idx}.jpg'
            )

        torchvision.utils.save_image(y.unsqueeze(1).float(), f'{folder}/y_{idx}.jpg')

    model.train()


def check_accuracy_per_class(loader, model, device="cuda"):
    # num_correct = 0
    # num_pixels = 0
    # acc_per_class = [0, 0, 0]
    # dice_score_per_class = [0, 0, 0]
    # precision_per_class = [0, 0, 0]

    # model.eval()
    # # 

    # with torch.no_grad():
    #     for x, y in loader:
    #         x = x.to(device)
    #         y = y.to(device)
            
    #         with torch.amp.autocast(device_type=device, dtype=torch.float16):
    #             preds = torch.sigmoid(model(x))
            
    #         preds = (preds > 0.5).float()
    #         num_correct += (preds == y).sum()
    #         num_pixels += torch.numel(preds)

    #         dice_score_per_class += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    #         true_positives = ((preds == 1) & (y == 1)).sum().item()
    #         false_positives = ((preds == 1) & (y == 0)).sum().item()

    #         precision_per_class += true_positives / (true_positives + false_positives + 1e-8)

    # acc_per_class = num_correct / len(loader.dataset)
    # precision_per_class = torch.tensor(precision_per_class) / len(loader.dataset)
    # dice_score_per_class = torch.tensor(dice_score_per_class) / len(loader.dataset)

    # print(
    #     f'Got accuracy per pixel {num_correct/num_pixels*100:.2f}%'
    # )

    # class_names = ['Skin', 'Nose', 'Hair']
    # for c, name in enumerate(class_names):
    #     print(f'Accuracy for {name}: {acc_per_class[c].item() * 100:.2f}%')
    #     print(f'Dice score for {name}: {dice_score_per_class[c].item():.2f}')
    #     print(f'Precision for {name}: {precision_per_class[c].item():.2f}')

# def check_metrics(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0

    # accumulators per class
    acc_correct = [0, 0, 0]
    acc_total   = [0, 0, 0]

    dice_num = [0, 0, 0]
    dice_den = [0, 0, 0]

    prec_num = [0, 0, 0]
    prec_den = [0, 0, 0]

    iou_n = [0, 0, 0]   # numerator
    iou_d = [0, 0, 0]   # denominator

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)   # y already has shape [B, 3, H, W]

            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                preds = torch.sigmoid(model(x))

            preds = (preds > 0.5).float()

            # total accuracy across all pixels
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)

            # loop over 3 classes
            for c in range(3):
                preds_c = preds[:, c, :, :]
                y_c = y[:, c, :, :]

                # per-class accuracy
                acc_correct[c] += (preds_c == y_c).sum().item()
                acc_total[c]   += preds_c.numel()

                # Dice numerator/denominator
                intersection = (preds_c * y_c).sum().item()
                union = preds_c.sum().item() + y_c.sum().item()

                dice_num[c] += 2 * intersection
                dice_den[c] += union

                # Precision numerator/denominator
                true_pos = intersection
                false_pos = (preds_c.sum().item() - intersection)
                prec_num[c] += true_pos
                prec_den[c] += (true_pos + false_pos)

                # IoU
                iou_n[c] += intersection
                iou_d[c] += (preds_c.sum().item() + y_c.sum().item() - intersection)

    # ===== Final metrics =====
    class_names = ["Skin", "Nose", "Hair"]
    print('-----------------------------------------------')
    print(f"Overall pixel accuracy: {num_correct / num_pixels * 100:.2f}%\n")

    for c, name in enumerate(class_names):
        acc = acc_correct[c] / acc_total[c]
        dice = dice_num[c] / (dice_den[c] + 1e-8)
        precision = prec_num[c] / (prec_den[c] + 1e-8)
        iou_per_class = iou_n[c] / (iou_d[c] + 1e-8)

        print(f"{name}:")
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  Dice     : {dice:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  IoU      : {iou_per_class:.4f}\n")

    # Mean across classes
    mean_acc = sum(acc_correct) / sum(acc_total)
    mean_dice = sum(dice_num) / (sum(dice_den) + 1e-8)
    mean_prec = sum(prec_num) / (sum(prec_den) + 1e-8)

    print("Mean over classes:")
    print(f"  Accuracy : {mean_acc*100:.2f}%")
    print(f"  Dice     : {mean_dice:.4f}")
    print(f"  Precision: {mean_prec:.4f}\n")
    print('-----------------------------------------------')
    model.train()
