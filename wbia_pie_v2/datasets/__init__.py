# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
from .animal_datasets import WhaleShark, WhaleSharkCropped, MantaRayCropped
from .animal_datasets import GrayWhale
from .animal_datasets import HyenaBothsides
from .animal_datasets import WildHorseFace
# from .animal_datasets import Zebra
# from .animal_datasets import Cachalot, CachalotSubsample
from .animal_datasets import RW_Laterals
from .animal_datasets import ConfigDataset
from .animal_wbia import AnimalNameWbiaDataset  # noqa: F401


__image_datasets = {
    'whaleshark': WhaleShark,
    'whaleshark_cropped': WhaleSharkCropped,
    'mantaray_cropped': MantaRayCropped,
    'graywhale': GrayWhale,
    'hyena_bothsides': HyenaBothsides,
    'wildhorse_face': WildHorseFace,
    # 'zebra': Zebra,
    # 'cachalot': Cachalot,
    # 'cachalot_subsample': CachalotSubsample,
    'rw_laterals': RW_Laterals,
    'config_dataset': ConfigDataset,
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
