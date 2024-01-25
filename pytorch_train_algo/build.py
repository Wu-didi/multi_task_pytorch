
from __future__ import absolute_import, print_function
futureimport absolute import,print functionfrom
from registry_config.utils import _as_list, Merge_model_export,get_unshared_and_shared_params
from registry_config.logger import logger
import pprint
from registry config.hooks import (
    DoCheckPoint,
    TimerHook,
    LogLR
    )
from registry_config.estimator import ComposeEstimator, Estimator, GradCollectTypeimport copy
import warnings
from easydict import EasyDict
from registry_config.lr scheduler import build lr scheduler
from registry_config.model zoo.build import build block
from registry_config.initializer import build initializer
from registry_config.data.build import build data loader
from registry_config.batch processor import build batch processor
from registry_config.metric import build metric
from registry_config.optimizer import build optimizer
import torch
import os
import tempfile
import torch.nn as nn

class _UnsetFlag:
""""
Special flag for unset values .
"""
pass

unset_flag = _UnsetFlag()

class DefaultTaskBuilder(object):
        '''
        Task builder, building objects from config.

        Parameters

        cfg : :py:class: auto_matrix.config. Config
        Data loader, model, etc.
        env_cfg : :py:class: auto_matrix. config. Config
        Environment variables, such as BPU March, kvstore
        solver_cfg : :py:class: auto_matrix.config. Config
        Solver config.
        step : str
        Current training step.
        '''

    def __init__(self, cfg, solver_cfg):
        self.cfg = cfg
        self.solver_cfg = solver_cfg
        self.logger = logger

        self. model = unset_flag
        self._export_model = unset_flag
        self._val_model = unset_flag
        self._initializer = unset_flag
        self._data_loader = unset_flag
        self._val_data_loader = unset_flag
        self._batch_prosessor = unset_flag
        self._val_batch_processor = unset_flag
        self._metrics = unset_flag
        self._metrics_and_comparator_pairs = unset_flag
        self._val_metrics = unset_flag
        self._hooks = unset_flag
        self._export_flags = unset_flag
        self._split_export_model = unset_flag
        self._split_export_flags = unset_flag

        self._temp_model_params = None

    def __del__(self):
        self._data_loader = unset_flag
        self._val_data_loader = unset_flag
        if self._temp_model_params is not None:
            os.remove(self._temp_model_params)

    def _build_model(self, name, cfg, solver_cfg=None):
        if solver_cfg is None:
            solver_cfg = self.solver_cfg
        model = build_block(name, cfg)
        assert isinstance(model, torch.nn.Module)
        return model

    def _reload_train_model_params(self):
        assert self ._ temp_model_params is not None, 'Please initialize self.model first' # noqa
        self. model. load_state_dict(torch. load(self ._ temp_model_params), strict=self.solver_cfg.get('allow_missing', True))

    def _build(self, build_name, cfg, *args, ** kwargs):
        name2build = {
            'model': self ._ build_model,
            'initializer': build_initializer,
            'data_loader': build_data_loader,
            'metrics': build_metric,
            'batch_processor':build_batch_processor,}

        build_fn = name2build.get(build_name, None)
        assert build_fn is not None, 'Unsupported build name %s' %build_name
        return build_fn(cfg.type, cfg, *args, ** kwargs)

    @property
    def initializer(self):
        if self. initializer is unset_flag:
            self. initializer = self. build(
            'initializer', self.cfg.initializer)

        assert callable(self ._ initializer), \
                 'self. initializer should callable'
        return self. initializer

    @property
    def model(self):
        if self._model is unset_flag:
            self._model = self ._ build('model', self.cfg.model)
            for parameter,weight in self ._ model.state_dict().items():
                if weight.dim()>2 and 'loss' not in parameter:# 针对bn结构
                    self.initializer(weight)

            if self.solver_cfg.load_model_path is not None:
                load_model_path = self.solver_cfg. load_model_path
                assert load_model_path. endswith('.pth')
                self. model.load_state_dict(torch.load(load_model_path), strict=self.solver_cfg.get('allow_missing', True))
    return self. model

    @property
    def val_model(self):
        if self ._ val_model is unset_flag:
            if self.cfg.get('val_model', None) is None:
                 self ._ val_model = self. build('model', self.cfg.model)
            else:
                 self ._ val_model = self. build('model', self.cfg. val_model)

            self ._ reload_train_model_params()

        return self ._ val_model

    @property
    def export_model(self):
        if self._export_model is unset_flag:
            if self.cfg.get('export_model', None) is None:
                if self.cfg.get('val_model', None) is None:
                    self._export_model = self._build('model', self.cfg.model)
                else:
                    self._export_model = self._build('model', self.cfg.val_model)
            else:
                self._export_model =self._build('mdoel', self.cfg.export_model)

            self._reload_train_model_params()

            if isinstance(self._export_model, torch.nn.Module):
                self.initializer(self._export_model)
                self._export_model.cpu()
        else:    
            msg = 'export_model is not HybridBlock, ignored ... '
            warnings.warn(msg)
            self._export_model = None

        return self ._export_model

    @property
    def data_loader(self):
        if self. data_loader is unset_flag:
            self. data_loader = self. build(
                'data_loader', self.cfg.data_loader
            )
        return self._data_loader

    @property
    def val_data_loader(self):
        if self ._ val_data_loader is unset_flag:
            if self.cfg.get('val_data_loader', None) is None:
                self._val_data_loader = None
            else:
                self._val_data_loader = self. build(
                'data_loader', self.cfg.val_data_loader
                )
        return self._val_data_loader

    @property
    def hooks(self): # TODO
        if self._hooks is unset_flag:
            self._hooks = []

        return self._hooks

    @property
    def metrics(self):
        if self._metrics is unset_flag:
            self._metrics = []
            for cfg_i in _as_list(self.cfg.metrics):
                cfg_i = copy.copy(cfg_i)
                cfg_i.pop('comparator', None)
                self._metrics.append(self._build('metrics', cfg_i))
        return self._metrics

    @property
    def val_metrics(self):
        if self._val_metrics is unset_flag:
            if self.cfg.get('val_metrics', None) is None:
            cfg = self.cfg.metrics
        else:
            cfg = self.cfg.val_metrics
            self._val_metrics = [self._build('metrics', cfg_i) for cfg_i in _as_list(cfg)]

        return self._val_metrics

    @property
    def batch_processor(self):
        if self._batch_prosessor is unset_flag:
            self._batch_prosessor = self._build('batch_processor', self.cfg.batch_processor,
            need_backward=True)
        return self._batch_prosessor

    @property
    def val_batch_processor(self):
        if self._val_batch_processor is unset_flag:
            if self.cfg.get('val_batch_processor', None) is None:
                self._val_batch_processor = self._build(
                'batch_processor', self.cfg.batch_processor,
                need_backward=False)
        else:
            self ._ val_batch_processor = self. build(
                'batch_processor', self.cfg.val_batch_processor,
                need_backward=False)
        return self ._ val_batch_processor

    @property
    def batch_size(self):
        _batch_size = sum(_as_list(self.cfg.data_loader.batch_size))
        return batch_size

    @property
    def batch_size_per_ctx(self):
        return int(self.batch_size)

    @property
    def export_flags(self):
        if self._export_flags is unset_flag:
            flags = dict(
                remove_internal_node_in_outputs=True,
                )
            flags.update(self.cfg.get('export_flags', dict()))
            self._export_flags = EasyDict(flags)
        return self._export_flags


class DefaultTrainer(object):
    def __init__(self,cfg, solver_cfg, task_builder):
        self.cfg = cfg
        self.solver_cfg = solver_cfg
        self.task_builders = _as_list(task_builder)
        self.params = None
        self.logger = logger
        self.hooks = []
        self._is_prepare = False
        self._optimizer = unset_flag
        self._lr_scheduler = unset_flag
        self.estimator = unset_flag



    def prepare(self, enable_export_ckpt_only):
        self ._ is_prepare = True
        models = []

        for builder_i in self.task_builders:
            models.append(builder_i.model)
            if self.params is None:
            self.params = models[-1].state_dict()
            else:
            self.params.update(models[-1]. state_dict())

        if len(models) > 1:
            unshared_params, shared_params = get_unshared_and_shared_params(
            [model_i.state_dict() for model_i in models])
            self.logger.info('%d shared parameters: \n%s\n' % (
            len(shared_params), pprint. pformat(shared_params)))
            self.logger.info('%d unshared parameters: \n%s\n' % (
            len(unshared_params), pprint. pformat(unshared_params)))

        self.hooks = [
            TimerHook(logger=self.logger),
            LogLR(frequent=self.cfg.env.log_interval,logger=self.logger)
        ]
        estimators = []
        assert len(self.task_builders) == len(models)
        for idx, (builder_i,model_i) in enumerate(zip(self.task_builders, models)):
            estimator_i= Estimator(
                            network=model_i,
                            data_loader=builder_i.data_loader,
                            batch_processor=builder_i.batch_processor,
                            metrics = builder_i.metrics,
                            hooks = builder_i.hooks,
                            logger = self.logger)
            estimators.append(estimator_i)
            # self.hooks. insert(2, DocheckPoint(
            # models=models,
            # enable_before_run=True,
            # prefix = self.solver_cfg. save_model_prefix,
            # logger = self.logger,
            # period = self.cfg.env.get('save_chckpint_iter_period',-1)
            # ))
            self.estimator = ComposeEstimator(
                    estimators=estimators, task_sample = self.cfg.get("TASK_SAMPLER", None),
                    logger=self.logger,
                    hooks=self.hooks)

    @ property
    def optimizer(self):
        if self._optimizer is unset_flag:
            opt_params = copy.deepcopy(self.solver_cfg.optimizer)
            opt_type = opt_params.pop('type')
            # backward compatible
            flag = opt_params.pop('auto_rescale_lr', None)
            self.parameters = []
            params_dict = dict()
            for builder_i in self.task_builders:
                for layer in builder_i.model.named_parameters():
                    params_dict.update({layer[0]:layer[1]})
            for _,value in params_dict.items():
                self.parameters.append(value)
            opt_params[ 'params' ] = self.parameters
            if flag is not None:
                msg = "auto_rescale_lr' is deprecated in optimizer, ' \
                    'please implement the automatically rescale learning
                    'rate strategy in config directly, this parameter is
                    'ignored for backward compatible'
                warnings.warn(msg)
            self._optimizer = EasyDict(
                                type=opt_type,
                                params=opt_params
                                )
            self.solver_cfg.lr_scheduler.optimizer = \
                build_optimizer(self._optimizer.type,
                                self._optimizer.params)
            self.optimizer_build = build_optimizer(self._optimizer.type,
                                                        self._optimizer.params)
        return self.optimizer_build, self._optimizer    

    @ property
    def lr_scheduler(self):
        self.optimizer_build, self. optimizer = self.optimizer
        if self._lr_scheduler is unset_flag:
            lr_scheduler = build_lr_scheduler(
                self.solver_cfg.lr_scheduler.type,
                self.solver_cfg.lr_scheduler)

            if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                opt_lr = self._optimizer['params']['lr']
            self._lr_scheduler = lr_scheduler
        return self._lr_scheduler

    def fit(self):
        assert self._is_prepare, \
            'call prepare() first'

        def _get_grad_collect_type():
            if len(self.task_builders) == 1:
                return GradCollectType.kByGradReqWrite
            else:
                return GradCollectType.kByManualCollectAndAdd
        start_epoch = 0
        start_iter = 0

        self.estimator.fit(
            trainer=self.params,
            ctx = self.cfg.env.ctx,
            num_epochs = -1,
            optimizer = self.optimizer[0],
            lr_scheduler=self.lr_scheduler,
            start_epoch=start_epoch,
            start_iter=start_iter,
            num_iters=self.solver_cfg.num_iter,
            grad_collect_type =_ get_grad_collect_type()
            )

    def end_training(self):
        assert self._is_prepare, 'call prepare() first
        for hook_i in self.hooks:
            if isinstance(hook_i,DoCheckPoint):
                hook_i.before_run()
                hook_i.after_run()