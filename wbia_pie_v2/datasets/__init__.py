# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
from .animal_datasets import WhaleShark, WhaleSharkCropped, MantaRayCropped
from .animal_datasets import GrayWhale, Hyena, HyenaLeftsNoval, HyenaBothsides
from .animal_wbia import AnimalNameWbiaDataset  # noqa: F401


__image_datasets = {
    'whaleshark': WhaleShark,
    'whaleshark_cropped': WhaleSharkCropped,
    'mantaray_cropped': MantaRayCropped,
    'graywhale': GrayWhale,
    'hyena': Hyena,
    'hyena_lefts_noval': HyenaLeftsNoval,
    'hyena_bothsides': HyenaBothsides,

}


def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    avai_datasets = list(__image_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __image_datasets[name](**kwargs)
