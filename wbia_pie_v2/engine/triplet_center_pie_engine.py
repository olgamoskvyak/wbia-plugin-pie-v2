# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import metrics
from losses import TripletLoss, CrossEntropyLoss, CenterLoss

from engine import PIEEngine


class TripletCenterPIEEngine(PIEEngine):
    r"""Triplet-loss engine for image-reid.

    Args:
        datamanager (DataManager): an instance of datamanager.
        model (nn.Module): model instance.
        optimizer (Optimizer): an Optimizer.
        margin (float, optional): margin for triplet loss. Default is 0.3.
        weight_t (float, optional): weight for triplet loss. Default is 1.
        weight_x (float, optional): weight for softmax loss. Default is 1.
        scheduler (LRScheduler, optional): if None, no learning rate decay.
        use_gpu (bool, optional): use gpu. Default is True.
        label_smooth (bool, optional): use label smoothing regularizer.
                Default is True.
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        margin=0.3,
        weight_t=1,
        weight_x=1,
        weight_c=1,
        scheduler=None,
        use_gpu=True,
        label_smooth=True,
    ):
        super(TripletCenterPIEEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        assert weight_t >= 0 and weight_x >= 0
        assert weight_t + weight_x > 0
        self.weight_t = weight_t
        self.weight_x = weight_x
        self.weight_c = weight_c

        self.criterion_t = TripletLoss(margin=margin)
        self.criterion_x = CrossEntropyLoss(
            num_classes=self.datamanager.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth,
        )
        # TODO add feat dim
        self.criterion_c = CenterLoss(
            num_classes=self.datamanager.num_train_pids,
            feat_dim=512,
            use_gpu=self.use_gpu,
        )
        print('***Initialized Triplet Center PIE Engine***')

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)

        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        outputs, features = self.model(imgs)

        loss = 0
        loss_summary = {}

        if self.weight_t > 0:
            loss_t = self.compute_loss(self.criterion_t, features, pids)
            loss += self.weight_t * loss_t
            loss_summary['loss_t'] = loss_t.item()

        if self.weight_c > 0:
            loss_c = self.compute_loss(self.criterion_c, features, pids)
            loss += self.weight_c * loss_c
            loss_summary['loss_c'] = loss_c.item()

        if self.weight_x > 0:
            loss_x = self.compute_loss(self.criterion_x, outputs, pids)
            loss += self.weight_x * loss_x
            loss_summary['loss_x'] = loss_x.item()
            loss_summary['acc'] = metrics.accuracy(outputs, pids)[0].item()

        assert loss_summary

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_summary
