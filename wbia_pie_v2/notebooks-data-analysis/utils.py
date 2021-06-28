# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import imageio as io
import math


def plot_names(
    names,
    filenames,
    bboxes,
    display_labels,
    crop=False,
    max_display=16,
    ncols=4,
    ratio=(4, 4),
    name=None,
):
    if name is None:
        # Select random name
        name_idx = np.random.choice(len(names), size=1)[0]
        name = names[name_idx]

    assert name in names

    num_samples = (names == name).sum()
    print('Found {} annots for name {}'.format(num_samples, name))

    # Get all samples for selected name
    sel_filenames = filenames[names == name][:max_display]
    sel_bboxes = bboxes[names == name][:max_display]
    sel_display_labels = display_labels[names == name][:max_display]

    nrows = math.ceil(min(num_samples, max_display) / ncols)
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ratio[0] * ncols, ratio[1] * nrows)
    )

    if nrows == 1:
        for c in range(ncols):
            index = c
            if index < num_samples:
                image = io.imread(sel_filenames[index])
                x, y, w, h = sel_bboxes[index]
                if crop:
                    image = image[y : y + h, x : x + w]
                ax[c].imshow(image)
                if not crop:
                    ax[c].plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], '-r')
                ax[c].set_title('{}'.format(sel_display_labels[index]))
            ax[c].axis('off')
    else:
        for r in range(nrows):
            for c in range(ncols):
                index = r * ncols + c
                if index < num_samples:
                    image = io.imread(sel_filenames[index])
                    x, y, w, h = sel_bboxes[index]
                    if crop:
                        image = image[y : y + h, x : x + w]
                    ax[r, c].imshow(image)
                    if not crop:
                        ax[r, c].plot(
                            [x, x + w, x + w, x, x], [y, y, y + h, y + h, y], '-r'
                        )
                    ax[r, c].set_title('{}'.format(sel_display_labels[index]))
                ax[r, c].axis('off')

    plt.tight_layout()
