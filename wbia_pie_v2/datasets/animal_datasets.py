# -*- coding: utf-8 -*-
from .coco_dataset import COCODataset


class WhaleShark(COCODataset):
    def __init__(self, **kwargs):
        super(WhaleShark, self).__init__(
            name='whaleshark',
            dataset_dir='whaleshark',
            dataset_url='https://lilablobssc.blob.core.windows.net/whale-shark-id/whaleshark.coco.tar.gz',
            split='train2020',
            crop=True,
            resize=True,
            imsize=256,
            train_min_samples=5,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['left', 'right'],
            debug=False,
            **kwargs
        )


class WhaleSharkCropped(COCODataset):
    def __init__(self, **kwargs):
        super(WhaleSharkCropped, self).__init__(
            name='whaleshark_cropped',
            dataset_dir='whaleshark_cropped',
            dataset_url='https://www.dropbox.com/s/4tky8z0g4ob6qfx/coco.whaleshark_cropped.tar.gz?dl=1',
            split='test2021',
            viewpoint_csv='reid-data/whaleshark_cropped/original/whaleshark_cropped.coco/annotations/annotViewpointMap.csv',
            crop=False,
            resize=True,
            imsize=256,
            train_min_samples=5,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['left', 'right'],
            debug=False,
            **kwargs
        )


class MantaRayCropped(COCODataset):
    def __init__(self, **kwargs):
        super(MantaRayCropped, self).__init__(
            name='mantaray_cropped',
            dataset_dir='mantaray_cropped.coco',
            dataset_url=None,
            split='train2021',
            crop=False,
            resize=False,
            imsize=300,
            train_min_samples=5,
            id_attr=['name'],
            debug=False,
            **kwargs
        )


class GrayWhale(COCODataset):
    def __init__(self, **kwargs):
        super(GrayWhale, self).__init__(
            name='graywhale',
            dataset_dir='graywhale',
            dataset_url='',
            split='train_right',
            split_test='test_left',
            crop=True,
            flip_test=True,
            resize=True,
            imsize=256,
            train_min_samples=3,
            test_min_samples=3,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['left', 'right'],
            debug=False,
            excluded_names='____',
            **kwargs
        )


class WildHorseFace(COCODataset):
    def __init__(self, **kwargs):
        super(WildHorseFace, self).__init__(
            name='wildhorse_face',
            dataset_dir='wildhorses_combined',
            dataset_url='',
            split='train2021',
            crop=True,
            flip_test=False,
            resize=True,
            imsize=300,
            train_min_samples=3,
            id_attr=['name', 'viewpoint'],
            viewpoint_list=['front'],
            debug=False,
            excluded_names='____',
            **kwargs
        )
