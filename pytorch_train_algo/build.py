
from __future__ import absolute_import, print_function # Python 2/3 compatibility
from registry_config.utils import _as_list, Merge_model_export,get_unshared_and_shared_params
from registry_config.logger import logger
import pprint
from registry_config.hooks import (
    DoCheckPoint,
    Evaluate,
    LogMeric,
    TimerHook,
    LogLR
    )
from registry_config.estimator import ComposeEstimator, Estimator, GradCollectTypeimport 
import warnings
from easydict import EasyDict
# from registry_config.lr scheduler import build lr scheduler
# from registry_config.model zoo.build import build block
# from registry_config.initializer import build initializer
# from registry_config.data.build import build data loader
# from registry_config.batch processor import build batch processor
# from registry_config.metric import build metric
# from registry_config.optimizer import build optimizer
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

    def __init__(self, cfg, env_cfg, solver_cfg):
        pass


def DefaultTrainer(object):
    def __init__(self, cfg, solver_cfg, task_builder):
        pass
