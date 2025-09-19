import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import os

def get_img_norm_cfg(dataset_name, dataset_dir):
    if dataset_name == 'SIRST':
        img_norm_cfg = dict(mean=101.06385040283203, std=34.619606018066406)
    elif dataset_name == 'NUDT-SIRST':
        img_norm_cfg = dict(mean=107.80905151367188, std=33.02274703979492)
    elif dataset_name == 'IRSTD-1k':
        img_norm_cfg = dict(mean=87.4661865234375, std=39.71953201293945)
    return img_norm_cfg

def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg['mean']) / img_norm_cfg['std']

def PadImg(img, times=32):
    h, w = img.shape
    if not h % times == 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if not w % times == 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img

class TestSetLoader(Dataset):
    def __init__(self, args, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.args = args
        self.dataset_dir = args.datasetpath + '/' + args.dataset
        # with open(self.dataset_dir + '/default/test.txt', 'r') as f:
        self.suffix = '.png'
        
        self.imgs_dir = os.path.join(args.datasetpath, args.dataset, 'images')
        self.label_dir = os.path.join(args.datasetpath, args.dataset, 'masks')
        self.train_txt = os.path.join(args.datasetpath, args.dataset, 'test.txt')
        with open(self.train_txt, "r") as f:
            self.img_names = f.read().splitlines()
            f.close()
        
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(args.dataset, self.dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg

    def __getitem__(self, idx):
        name = self.img_names[idx]
        
        img_path = os.path.join(self.imgs_dir, name + self.suffix)
        label_path = os.path.join(self.label_dir, name + self.suffix)
        
        img = Image.open(img_path).convert('I')
        mask = Image.open(label_path)
        
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        h, w = img.shape

        img = PadImg(img)
        
        img, mask = img[np.newaxis, :], mask[np.newaxis, :]

        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h, w], self.img_names[idx]

    def __len__(self):
        return len(self.img_names)