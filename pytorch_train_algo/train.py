import argparse
import pprint
import sys
sys.path.append('./')
from pytorch_train_algo.build import DefaultTaskBuilder, DefaultTrainer
import os
import torch   
import time
from registry_config.config import Config
from registry config.utils import _as_list
import copy
from registry_config.logger import logger

def parse args():
    parser = argparse.ArgumentParser()
    parser.add argument('--config',type=str,required=True.help='configure file')
    parser.add argument('--ctx',type=str,default=None)
    parser.add argument('--num-machines ',type=int,default=1.help='The number of machines')
    parser.add argument('--export-ckpt-only',action='store true',
                                            default=False,
                                            help='skip trainer.fit(),export checkpoints only')
    args = parser.parse args()
    
    assert args.num machines >= 1,'The number of machines is at least 1
    
    return args



def setup_config(args):
    def _pop_unused_keys(configs):
        used_keys = [
            'BASE_CONFIG', 'TASK_CONFIGS',
            'model', 'val_model',
            'data_loader', 'val_data_loader',
            'initializer', 'batch_processor', 'val_batch_processor',
            'metrics', 'val_metrics', 'solver', 'env',
            'allow_use_subset_of_params',
            'max_input_shape', 'min_input_shape',]

        for config_i in _as_list(configs):
            unused_keys = []
            for key_i in config_i.keys():
                if key_i not in used_keys:
                    unused_keys.append(key_i)
            for key_i in unused_keys:
                config_i.pop(key_i)
    config = Config. load_file(args.config)
    _pop_unused_keys(config)

    if config.get('TASK_CONFIGS',None) is not None:
        task_configs = []
        for config_i in _as_list(config.TASK_CONFIGS):
            config_i = os.path.join(os.path.dirname(args.config), config_i)
            config_i = Config.load_file(config_i)
            _pop_unused_keys(config_i)
            task_configs.append(config_i)
        config. TASK_CONFIGS = task_configs
    else:
        task_configs = [copy.copy(config) ]

    def _get_solver_config(config):
    if isinstance(config.solver,dict):
        solver_config = config.solver
        return solver_config
    else:
        raise TypeError("config.solver should be a dict not %s"%(type(config.solver)))

    solver_config = _get_solver_config(config)

    for task_config_i in task_configs:
        for key in ['solver', 'env', 'allow_use_subset_of_params',
                    'ignore_wd_except_weights']:
            task_config_i.pop(key, None)
            # task_config_i.set_immutable()
        # solver_config.set_immutable()
        # config.set_immutable()

    return config, task_configs, solver_config

def main():
    args = parse_args()
    config, task_configs, solver_config = setup_config(args)
    logger.info('=' * 50+'BEGIN TRAINING MODE'+'='*50)
    logger. info('Training with config: \n%s',pprint. pformat(config))

    task_builders = [DefaultTaskBuilder(task_config_i, solver_config)
                     for task_config_i in task_configs]
    trainer = DefaultTrainer(config, solver_config, task_builders)
    trainer. prepare(enable_export_ckpt_only=args.export_ckpt_only)
    if args.export_ckpt_only:
        trainer.end_training()
    else:
        trainer.fit()


    logger.info('='*50 + 'END TRAINING MODE ' + '='*50)

if __name__ == '__main__':
    main()