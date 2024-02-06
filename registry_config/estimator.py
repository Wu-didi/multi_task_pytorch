from __future__ import absolute_import, print_function
from typing import Any # Python 2/3 compatibility
from registry_config.logger import logger
from easydict import EasyDict 
from collections import OrderedDict
from registry_config.utils import _as_list, _reset_dataloader
from registry_config.hooks import Hook
import torch
from registry_config.misc import Perf

class GradCollectType(object):
    """
    determine the type of gradient collection
    
    """
    kByGradReqAdd = 'kByGradReqAdd'
    kByManualCollectAndAdd = 'kByManualCollectAndAdd'
    kByGradRepWrite = 'kByGradRepWrite'

class BaseEstimator(object):
    def __init__(self, hooks = None)
        self.hooks = []
        if hooks is not None:
            self.register_hooks(hooks)
    
    def register_hooks(self, hooks):
        for hook in _as_list(hooks):
            assert isinstance(hook, Hook)
            self.hooks.append(hook)
    
    def before_run(self, locals):
        for hook in self.hooks:
            hook.before_run(locals)

    def after_run(self, locals):
        for hook in self.hooks:
            hook.after_run(locals)

    def before_epoch(self, locals):
        for hook in self.hooks:
            hook.before_epoch(locals)

    def after_epoch(self, locals):
        for hook in self.hooks:
            hook.after_epoch(locals)

    def before_iter(self, locals):
        for hook in self.hooks:
            hook.before_iter(locals)
    
    def after_iter(self, locals):
        for hook in self.hooks:
            hook.after_iter(locals)
    
    def fit(self, *args, **kwargs):
        raise NotImplementedError

class Estimator(BaseEstimator):
    def __init__(self,
                 network,
                 data_loader,
                 batch_processor, 
                 metrics,
                 logger = logger,
                 hooks = None):
        self.network = network
        self.data_loader = data_loader
        self.metrics = []
        if metrics is not None:
            self.metrics = _as_list(metrics)
        if len(self.metrics) == 1 and \
            not isinstance(metrics, (list, tuple)):
            self.metrics = self.metrics[0]
        self.batch_processor = batch_processor
        assert callable(self.batch_processor), \
            'batch_processor must be callable'
        self.logger = logger
        super().__init__(hooks)

    def fit(self, trainer,
                  ctx,
                  num_epochs,
                  start_epoch = 0,
                  lr_scheduler = None,
                  log_perf_freq = 1000,
                  num_iters = None,
                  start_iter = 0):
        self.logger.info("Training with device: {}".format(ctx))
        self.logger.info("Start training from [Epoch {}], Iter [%d]".format(start_epoch, start_iter))
        assert num_epochs > 0 or num_iters > 0,\
            'num_epochs and num_iters can not be both None'
     
        if num_epochs > 0:
            assert num_epochs > start_epoch, \
                'num_epochs must be greater than start_epoch'
        if num_iters > 0:
            assert num_iters > start_iter, \
                'num_iters must be greater than start_iter'
        
        perf_metric = Perf('update', freq = log_perf_freq)

        batch_id = start_iter
        epoch_id = start_epoch
        end_of_training_flag = False
        if num_epochs > 0 and start_epoch == num_epochs:
            end_of_training_flag = True
        if num_iters > 0 and start_iter == num_iters:
            end_of_training_flag = True

        self.before_run(locals())

        while not end_of_training_flag:
            data_batch = next(iter(self.data_loader))
            nbatch = 0
            epoch_end_flag = False

            self.before_epoch(locals())

            while not epoch_end_flag:
                self.before_iter(locals())
                batch_outputs = self.batch_processor(
                    self.network, data_batch, ctx, self.metrics, locals()
                )

                self.after_iter(locals())
                try:
                    data_batch = next(iter(self.data_loader))
                except StopIteration:
                    epoch_end_flag = True

                nbatch += 1
                batch_id += 1

                if num_iters > 0 and batch_id + 1 >= num_iters:
                    end_of_training_flag = True
                    break

            self.after_epoch(locals())

            # update epoch_id
            epoch_id += 1

            # check if training should be stopped
            if num_epochs > 0 and epoch_id >= num_epochs:
                end_of_training_flag = True
        self.after_run(locals())
        self.logger.info("Training finished")

class ComposeEstimator(BaseEstimator):
    def __init__(self, estimators, tast_sample = None,
                  logger = logger, hooks = None):
        self.estimators = _as_list(estimators)
        assert self.num_estimators >= 1, 'at least one estimator is required'
        self.logger = logger
        self._should_reassign_params = tast_sample is not None
        _hooks = []

        for estimator_i in self.estimators:
            _hooks.extend(estimator_i.hooks)
        if hooks is not None:
            _hooks.extend(_as_list(hooks))
        super.__init__(_hooks)

    @property
    def num_eastimators(self):
        return  len(self.estimators)
    
    @property
    def network(self):
        return [estimator.network for estimator in self.estimators]
    
    @property
    def data_loader(self):
        return [estimator.data_loader for estimator in self.estimators]