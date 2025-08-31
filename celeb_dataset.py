import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CelebA_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, is_test = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.is_test = is_test

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        if self.is_test == True: index += 2000

        skin_path = os.path.join(self.mask_dir, f"{str(index).zfill(5)}_skin.png")
        nose_path = os.path.join(self.mask_dir, f"{str(index).zfill(5)}_nose.png")        
        
        hair_path = os.path.join(self.mask_dir, f"{str(index).zfill(5)}_hair.png")
        if not os.path.exists(hair_path): hair_path = None

        mask_path = [skin_path, nose_path, hair_path]

        # image
        image = np.array(Image.open(img_path).convert('RGB'))

        # mask preparation. Creates a 3 channels mask image
        mask = []
        for path in mask_path:
            if path is not None:
                m = np.array(Image.open(path).convert("L"), dtype=np.float32)
                m[m == 255.0] = 1.0
            else:
                m = np.zeros((512, 512), dtype=np.float32)  
            mask.append(m)
        mask = np.stack(mask, axis=-1)
        
        # transform
        if self.transform is not None:
            augmentations = self.transform(image = image, mask = mask)
            image = augmentations['image']
            mask = augmentations['mask']

        mask = np.transpose(mask, (2, 0, 1)) 

        return image, mask
