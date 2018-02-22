import numpy as np
import os

import chainer
from chainercv import utils

root = '/home/wei/'

def get_dir():
    return os.path.join(root, 'data')

class DefectDetectionDataset(chainer.dataset.DatasetMixin):
    """Base class for defect defection dataset
    """

    def __init__(self, data_dir='auto'):
        if data_dir == 'auto':
            data_dir = get_dir()

        self.data_dir = data_dir
        images_file = os.path.join(data_dir, 'images.txt')
        bbs_file = os.path.join(data_dir, 'bounding_boxes.txt')
        image_class_labels_file = os.path.join(
            data_dir, 'image_class_labels.txt')

        self.paths = [
            line.strip().split()[1] for line in open(images_file)]

        # bbs should be a matrix (m by 4). m is the number of bounding
        # boxes in the image
        # labels should be an integer array (m by 1). m is the same as the bbs

        # (x_min, y_min, width, height)
        bbs = np.array([
            tuple(map(float, line.split()[1:5]))
            for line in open(bbs_file)])
        # (x_min, y_min, width, height) -> (x_min, y_min, x_max, y_max)
        bbs[:, 2:] += bbs[:, :2]
        # (x_min, y_min, width, height) -> (y_min, x_min, y_max, x_max)
        bbs[:] = bbs[:, [1, 0, 3, 2]]
        self.bbs = bbs.astype(np.float32)
        labels = [int(d_label.split()[1]) - 1 for
                  d_label in open(image_class_labels_file)]
        # store class label as integer
        self._labels = np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.paths)

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
            os.path.join(self.data_dir, 'images', self.paths[i]),
            color=True)
        label = self._labels[i]

        return img, label, self.bbs[i]