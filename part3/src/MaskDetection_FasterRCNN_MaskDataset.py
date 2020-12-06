from PIL import Image
import os
import glob

from MaskDetection_FasterRCNN_utils import generate_target


class MaskDataset(object):
    def __init__(self, transforms, imgs_path, labels_path, mode):
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = []
        for filename in glob.glob(imgs_path + '/*.png'):
            self.imgs.append(filename)
        # self.imgs = list(sorted(os.listdir(imgs_path)))
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == "eval":
            idx = idx + 768
        # load images ad masks
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'
        img_path = os.path.join(self.imgs_path, file_image)
        label_path = os.path.join(self.labels_path, file_label)
        img = Image.open(img_path).convert("RGB")
        # Generate Label
        target = generate_target(idx, label_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target, img_path

    def __len__(self):
        return len(self.imgs)