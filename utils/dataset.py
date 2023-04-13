import glob
from pathlib import Path

from torch.utils.data.dataset import Dataset
import os
import yaml
from PIL import Image

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class MultiViewDataSet(Dataset):

    def __init__(self, root, set_type, transform):
        self.images = []
        self.labels = []
        self.root = root
        self.set_type = set_type
        self.transform = transform

        f = open(root, 'r')
        cfg = f.read()
        y = yaml.safe_load(cfg)

        self.num_cls = y['nc']
        self.classes = y['names']

        img_path = Path(y[self.set_type])
        if img_path.is_dir():
            t = glob.glob(str(img_path / '*.*'), recursive=True)
            self.images = sorted([x.replace('/', os.sep) for x in t if x.split('.')[-1].lower() in IMG_FORMATS])
        else:
            assert 'no data dir'

        self.labels = img2label_paths(self.images)

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = image.convert('RGB')
        image = self.transform(image)

        label_path = Path(self.labels[index])
        label = []
        if label_path.is_file():
            with open(label_path, 'r') as f:
                label = f.read().split(' ')
        else:
            assert 'no label file'
        return image, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.images)

