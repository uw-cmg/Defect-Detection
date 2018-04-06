from chainercv import transforms
import random


def rotate_bbox(bbox, size, k):
    """Rotate bounding boxes accordingly

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    :param bbox:
    :param size:
    :param k:
    :return:
    """
    H, W = size
    origin = (W/2, H/2)
    p1 = (bbox[:, 1], bbox[:, 0])
    p2 = (bbox[:, 3], bbox[:, 2])
    k = k % 4
    if k != 0 and len(p1) > 0:
        new_p1 = rotate_point(p1, origin, k)
        new_p2 = rotate_point(p2, origin, k)
        if new_p1[1][0] <= new_p2[1][0]:
            bbox[:, 0] = new_p1[1]
            bbox[:, 2] = new_p2[1]
        else:
            bbox[:, 0] = new_p1[1]
            bbox[:, 2] = new_p2[1]

        if new_p1[0][0] <= new_p2[0][0]:
            bbox[:, 1] = new_p1[0]
            bbox[:, 3] = new_p2[0]
        else:
            bbox[:, 1] = new_p1[0]
            bbox[:, 3] = new_p2[0]

    return bbox


def random_resize(img):
    rv = random.random()
    if rv < 0.5:
        ratio = round(rv*2, 1)
        _, H, W = img.shape
        img = transforms.resize(img, (int(ratio*H), int(ratio*W)))
    return img


def rotate_point(point, origin, k):
    x, y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = [1, 0, -1, 0][k]
    sin_rad = [0, 1, 0, -1][k]
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return qx, qy
