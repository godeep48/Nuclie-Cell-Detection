import albumentations as A
from albumentations.pytorch import ToTensor
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
import os
from skimage import io, transform
from torch.utils.data import Dataset



def preprocess_mask(mask):
    #mask = np.array(mask).astype(np.float32)
    mask[mask != 255.0] = 0.0
    mask[mask==255]=1.0
    return mask

def get_train_transform(image_size):
   return A.Compose(
       [
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        ToTensor()
        ])

class LoadDataSet(Dataset):
    def __init__(self, path, transform=None):
      #super().__init__()
      self.path = path
      self.images_folder = os.path.join(path, "images")
      self.masks_folder = os.path.join(path, "masks")
      self.transform = transform
      self.images_ids = os.listdir(self.images_folder)
        
    def __getitem__(self, idx):

      image_dir = [self.images_folder+"/"+ids for ids in self.images_ids]
      mask_dir = [self.masks_folder+"/"+ids[:-7]+"mask_buffered.png" for ids in self.images_ids]
      image_name = image_dir[idx]
      mask_name = mask_dir[idx]
      
      img = io.imread(image_name)[:,:,:3].astype('float32')
      mask = io.imread(mask_name).astype('float32')
      mask = preprocess_mask(mask)

      transformed = self.transform(image=img, mask=mask)
      img = transformed['image']
      mask = transformed['mask']
      #img = np.transpose(img, (2,0,1))
      #mask = np.transpose(mask, (2,0,1))

      return img, mask#, image_name

    def __len__(self) -> int:
        return len(self.images_ids)