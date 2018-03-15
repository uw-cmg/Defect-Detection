from .imageUtils import cropImage


def watershed_image(img):
    """
    use watershed flooding algorithm to extract the loop contour
    :param img: type(numpy.ndarray) image in CHW format
    :return: 
    """
