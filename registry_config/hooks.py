from __future__ import absolute_import, print_function # Python 2/3 compatibility
from collections import namedtuple
import torch.nn as nn
from registry_config.logger import logger
import time
from registry_config.utils import _as_list, Merge_model_export
from easydict import EasyDict 
from collections import OrderedDict
import torch
from torchmetrics import MetricCollection
import os
import copy

# ，你可以通过 batch_params.epoch 来获取训练周期的值。
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch','nbatch','eval_metric','locals'])

class Hook:

    def before_run(self, locals):
        pass

    def after_run(self, locals):
        pass

    def before_epoch(self, locals):
        pass

    def after_epoch(self, locals):
        pass

    def before_iter(self, locals):
        pass

    def after_iter(self, locals):
        pass


class TimerHook(Hook):
    ''' 
    get time while training
    '''

    def __init__(self,logger=logger):
        super(TimerHook,self).__init__()
        self.logger = logger
        self._run_tic = None
        self._epoch_tic = None

    def before_run(self,locals):
        self._run_tic = time.time()

    def after_run(self,locals):
        assert self._run_tic is not None
        self.logger.info('Total run time {:.2f}s'.format(
                            time.time()-self._run_tic))
        
    def before_epoch(self,locals):
        self._epoch_tic = time.time()

    def after_epoch(self,locals):
        assert self._epoch_tic is not None
        self.logger.info('Epoch[{}] Time cost {:.2f}s'.format(
                            locals['epoch'],time.time()-self._epoch_tic))
        

class LogLR(Hook):
    '''
    log learning rate
    '''
    def __init__(self,logger=logger):
        super(LogLR,self).__init__()
        self.frequent = 1
        self.logger = logger

    # print lr after some iter
    def after_iter(self,locals):
        count = locals['nbatch']
        batch_id = locals['batch_id']
        lr = locals['lr']
        epoch_id = locals['epoch_id']

        if count % self.frequent == 0:
            return 
        self.logger.info('Epoch[{}] Batch[{}] lr {:.4f}'.format(
                            epoch_id,batch_id,lr))
        
    
class DoCheckPoint(Hook):
    '''
    save model after epoch
    '''
    def __init__(self,models,prefix,abs_path,period=1,
                enable_before_run=True,logger=logger):
        super(DoCheckPoint,self).__init__()
        self.models = _as_list(models)
        self.prefix = prefix
        self.period = int(max(1,period))
        self.enable_before_run = enable_before_run
        self.logger = logger
        self.abs_path = abs_path

    def _do_checkpoint_after_epoch(self,pth_filename):
        params = {}
        for net_i in self.models:
            params.update(net_i.state_dict()) # update the dict
        torch.save(params,os.path.join(self.abs_path,pth_filename))
        self.logger.info('Saved checkpoint to {}'.format(pth_filename))

    # save model after the whole training, it used to save last model weight
    def _do_checkpoint_after_run(self,pth_filename):
        params = {}
        merge_model_param = copy.deepcopy(self.models)
        for idx,net_i in enumerate(merge_model_param):
            net_i.head._modules = OrderedDict([(str(idx),net_i.head._modules['0'])])
            params.update(net_i.state_dict())
        model_export = Merge_model_export(params,self.models)
        torch.save(model_export,os.path.join(self.abs_path,pth_filename))
        self.logger.info('Saved checkpoint to {}'.format(pth_filename))
        
    def after_epoch(self,locals):
        epoch_id = locals['epoch_id']
        if (epoch_id + 1) % self.period == 0:
            pth_filename = '{}-{:04d}.pth'.format(self.prefix,epoch_id+1)
            self._do_checkpoint_after_epoch(pth_filename)

    def after_run(self,locals):
        pth_filename = '{}-last.pth'.format(self.prefix)
        self._do_checkpoint_after_run(pth_filename)

    
