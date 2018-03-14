import matplotlib.pyplot as plt


def cropImage(img, bboxes):
    """crop images by the given bounding boxes.

    Args:
        img (numpy.ndarray): image in CHW format
        bboxes (numpu.ndarray): bounding boxes in the format specified by chainerCV

    Returns:
        a batch of cropped image in BCHW format
        The image is in CHW format and its color channel is ordered in
        RGB.

    Return type: numpy.ndarray

    """
    pass


def showImage(img):
    """
    :param img (numpy.ndarray): image in CHW format
    :return: plot the red channel in grayscale color map

    """
    plt.imshow(img.transpose((1, 2, 0))[:, :, 0], cmap='gray')