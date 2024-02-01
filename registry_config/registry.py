from typing import Any
from registry_config.logger import logger as LOGGER
import copy
import inspect
import warnings

class Registry(object):

    def __init__(self, name):
        self._name = name
        self._name_obj_map = {}

    def pop(self,name,default=None):
        return self._name_obj_map.pop(name,default)
    
    def _do_register(self, name, obj, allow_override, logger):
        if isinstance(name, str):
            name = name.lower()
        if name in self._name_obj_map and obj != self._name_obj_map[name]:
            if allow_override:
                logger.info("Overriding {} in {} with {}".format(name, self._name, obj))
            else:
                raise KeyError("{} is already registered in {}!".format(name, self._name))
        self._name_obj_map[name] = obj
        
    
    def register(self, obj, name = None, allow_override = False, logger = None):
        if name is None:
            name = obj.__name__
        logger = LOGGER if logger is None else logger
        self._do_register(name, obj, allow_override, logger)
        return obj
    
    def alias(self, name, allow_overwrite = False, logger = None):
        logger = LOGGER if logger is None else logger
        
        def reg(obj):
            self._do_register(name, obj, allow_overwrite, logger)
            return obj
        return reg
    
    def get(self, name):
        origin_name = name
        if isinstance(name, str):
            name = name.lower()
        ret = self._name_obj_map.get(name, None)
        if ret is None:
            raise KeyError("{} is not registered in {}!".format(origin_name, self._name))
        
        return ret
    
def default_legacy_build_fn_match_cond(build_fn):
    '''判断build_fn是否符合条件, 是否只有一个参数，且参数名为cfg, config, cfgs, configs'''
    # inspect.signature()返回一个inspect.Signature类型的对象，值为函数的参数签名
    params = list(inspect.signature(build_fn).parameters)
    return len(params) == 1 and params[0] in ['cfg', 'config', 'cfgs', 'configs']

def build_registry(registry, name, cfg,
                   legacy_build_fn_match_cond = default_legacy_build_fn_match_cond,
                   *args, **kwargs):
    '''构建注册表'''
    
    # 1. 从注册表中获取构建函数
    build_fn = registry.get(name)
    if callable(legacy_build_fn_match_cond) and legacy_build_fn_match_cond(build_fn):
        msg  = f'\033 [1;31mWARNING: \033[0m'
        warnings.warn(msg + f'Using legacy build function signature for {name}.')
        
        try:
            # 2. 构建函数的参数为cfg
            return build_fn(cfg, *args, **kwargs)
        except TypeError as e:
            # 3. 如果构建函数的参数不为cfg, 则报错
            raise TypeError('Build function {} does not have signature of (cfg, *args, **kwargs)'.format(name)) from e
        
    else:
        cfg = copy.deepcopy(cfg)
        if 'type' in cfg:
            # 4. 如果cfg中有type字段，则获取type字段的值
            cfg.pop('type')
        assert not args, 'Args is not supported now.'
        assert not kwargs, 'Kwargs is not supported now.'
        # 5. 返回构建函数的值
        try:
            return build_fn(cfg)
        except TypeError as e:
            raise TypeError('Build function {} does not have signature of (*args, **kwargs)'.format(name)) from e
            
            