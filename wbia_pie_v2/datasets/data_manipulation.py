# -*- coding: utf-8 -*-
# Written by Olga Moskvyak (olga.moskvyak@hdr.qut.edu.au)
import sys
import os.path as osp
from skimage import transform
import numpy as np
import math
import time
import torch
from PIL import Image


def increase_bbox(bbox, scale, image_size, type='xyhw'):
    """Increase the size of the bounding box
    Input:
        bbox_xywh:
        scale:
        image_size: tuple of int, (h, w)
        type (string): notation of bbox: 'xyhw' or 'xyx2y2'
    """
    if type == 'xyhw':
        x1, y1, bbox_w, bbox_h = bbox
        x2 = x1 + bbox_w
        y2 = y1 + bbox_h
    else:
        x1, y1, x2, y2 = bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1
    h, w = image_size

    increase_w_by = (bbox_w * scale - bbox_w) // 2
    increase_h_by = (bbox_h * scale - bbox_h) // 2

    new_x1 = int(max(0, x1 - increase_w_by))
    new_x2 = int(min(w - 1, x2 + increase_w_by))

    new_y1 = int(max(0, y1 - increase_h_by))
    new_y2 = int(min(h - 1, y2 + increase_h_by))

    if type == 'xyhw':
        return (new_x1, new_y1, new_x2 - new_x1, new_y2 - new_y1)
    else:
        return (new_x1, new_y1, new_x2, new_y2)


def to_origin(bbox_xywh, new_origin):
    """Update coordinates of bounding box after moving the origin
    Height and width do not change.
    Coordinates are allowed to be negative and go outside of image boundary.
    Input:
        coords (list of floats): coords of bbox, [x1, y1, w, h]

    """
    bbox_xywh[0] -= new_origin[0]
    bbox_xywh[1] -= new_origin[1]
    return bbox_xywh


def rotate_coordinates(coords, angle, rotation_centre, imsize, resize=False):
    """Rotate coordinates in the image"""
    rot_centre = np.asanyarray(rotation_centre)
    angle = math.radians(angle)
    rot_matrix = np.array(
        [
            [math.cos(angle), math.sin(angle), 0],
            [-math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    coords = transform.matrix_transform(coords - rot_centre, rot_matrix) + rot_centre

    if resize:
        rows, cols = imsize[0], imsize[1]
        corners = np.array(
            [[0, 0], [0, rows - 1], [cols - 1, rows - 1], [cols - 1, 0]], dtype=np.float32
        )
        if rotation_centre is not None:
            corners = (
                transform.matrix_transform(corners - rot_centre, rot_matrix) + rot_centre
            )

        x_shift = min(corners[:, 0])
        y_shift = min(corners[:, 1])
        coords -= np.array([x_shift, y_shift])

    return coords


def resize_coords(coords, original_size, target_size):
    """Resize coordinates
    Input:
        coords (list or tuple of floats): (x, y) coordinates
        original_size: size of image (h, w)
        target_size: target size of image (h, w)
    """
    assert isinstance(coords, (list, tuple, np.ndarray))
    assert len(coords) % 2 == 0
    assert len(original_size) == 2
    assert len(target_size) == 2

    if type(coords) == tuple:
        coords = list(coords)

    for i in range(0, len(coords), 2):
        coords[i] = int((coords[i] / original_size[1]) * target_size[1])
        coords[i + 1] = int((coords[i + 1] / original_size[0]) * target_size[0])
    return coords


def unnormalize(
    batch_image,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    use_gpu=False,
    convert_numpy=False,
):
    """Reverse normalization applied to batch of images
    Input:
        batch_image (torch Tensor): tensor of shape (bs, ch, h, w)"""
    B = batch_image.shape[0]
    H = batch_image.shape[2]
    W = batch_image.shape[3]
    t_mean = (
        torch.FloatTensor(mean)
        .view(3, 1, 1)
        .expand(3, H, W)
        .contiguous()
        .view(1, 3, H, W)
    )
    t_std = (
        torch.FloatTensor(std).view(3, 1, 1).expand(3, H, W).contiguous().view(1, 3, H, W)
    )
    if use_gpu:
        t_mean = t_mean.cuda()
        t_std = t_std.cuda()
    batch_image_unnorm = batch_image * t_std.expand(B, 3, H, W) + t_mean.expand(
        B, 3, H, W
    )
    if convert_numpy:
        batch_image_unnorm = batch_image_unnorm.numpy().transpose((0, 2, 3, 1))
    return batch_image_unnorm


def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib

    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            '\r...%d%%, %d MB, %d KB/s, %d seconds passed'
            % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write('\n')


def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(
                    path
                )
            )
    return img
