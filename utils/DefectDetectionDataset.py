import numpy as np
import os

import chainer
from chainercv import utils
from chainercv import transforms

root = '/home/wei/Data/Loop_detection/'



class DefectDetectionDataset(chainer.dataset.DatasetMixin):
    """Base class for defect defection dataset
    """

    def __init__(self, data_dir='auto', split='', img_size=1024, resize=False):
        if data_dir == 'auto':
            data_dir = root
        self.data_dir = data_dir
        self.img_size = img_size
        self.resize = resize
        images_file = os.path.join(self.data_dir, '{}images.txt'.format(split))
        self.images = [
            line.strip() for line in open(images_file)]

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and its label.
            The image is in CHW format and its color channel is ordered in
            RGB.
            a bounding box is appended to the returned value.
        """
        img = utils.read_image(
            os.path.join(self.data_dir, 'images', self.images[i]),
            color=True)
        # bbs should be a matrix (m by 4). m is the number of bounding
        # boxes in the image
        # labels should be an integer array (m by 1). m is the same as the bbs

        bbs_file = os.path.join(self.data_dir, 'bounding_boxes', self.images[i][0:-4]+'.txt')
        
        bbs = np.stack([line.strip().split() for line in open(bbs_file)]).astype(np.float32)
        label = np.stack([0]*bbs.shape[0]).astype(np.int32)

        _, H, W = img.shape
        if self.resize and (H != self.img_size or W != self.img_size):
            img = transforms.resize(img, (self.img_size, self.img_size))
            bbs = transforms.resize_bbox(bbs, (H, W), (self.img_size, self.img_size))

        return img, bbs, label