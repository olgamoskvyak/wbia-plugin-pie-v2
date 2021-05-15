# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import os.path as osp


def visualize_batch(
    batch, labels=None, save_dir='', figname='figure', ncols=4, figsize=(4, 4)
):
    """Visualise training batch

    batch (torch.Tensor): images of shape (B, C, H, W)
    """
    nrows = batch.size(0) // ncols

    assert nrows > 1
    assert ncols > 1

    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * figsize[0], nrows * figsize[1])
    )

    images = batch.detach().cpu()
    images = unnormalize(images, use_gpu=False)
    images = images.numpy().transpose((0, 2, 3, 1))

    for r in range(nrows):
        for c in range(ncols):
            index = r * ncols + c
            image = images[index]
            ax[r, c].imshow(image)
            if labels is not None:
                ax[r, c].set_title('{}'.format(labels[index]))
            ax[r, c].axis('off')

    # Save figure
    fig_path = osp.join(save_dir, figname + '.jpg')
    fig.savefig(fig_path, format='jpg', dpi=100, bbox_inches='tight', facecolor='w')
    plt.close(fig)


def unnormalize(
    batch_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], use_gpu=False
):
    """Reverse normalization applied to batch of images """
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
    return batch_image_unnorm
