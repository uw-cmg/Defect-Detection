import numpy as np
import os

import chainer
from chainercv import utils

root = '/home/wei/Data/Loop_detection/'

def get_dir():
    return root

class DefectDetectionDataset(chainer.dataset.DatasetMixin):
    """Base class for defect defection dataset
    """

    def __init__(self, data_dir='auto'):
        if data_dir == 'auto':
            data_dir = get_dir()

        self.data_dir = data_dir
        images_file = os.path.join(self.data_dir, 'images.txt')

        self.images = [
            line.strip() for line in open(images_file)]

    def __len__(self):
        return len(self.images)

    def get_example(self, i):
        """Returns the i-th example.

        Args:
            i (int): The index of the example.

        Returns:s
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
        label = np.stack([1]*bbs.shape[0]).astype(np.int32)

        return img, bbs, label