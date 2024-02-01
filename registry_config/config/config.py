from __future__ import absolute_import, print_function # Python 2/3 compatibility
import os
from config.loader import ext_to_load_fn_map


__all__ = ['Config']


class Config(dict):
    '''
    an wrapper of dict with easy attribute access and  attribute protection
    
    '''
    
    
    _BASE_CONFIG = 'BASE_CONFIG'
    _RECURSIVE_UPDATE_BASE_CONFIG = 'RECURSIVE_UPDATE_BASE_CONFIG'
    _MUTABLE = '_MUTABLE'
    
    
    @staticmethod
    def load_file(filename, all_unsafe=False):
        """
        GET CONFIG FROM FILE    

        Args:
            filename (_type_): _description_
            all_unsafe (bool, optional): _description_. Defaults to False.
        """
        ext = os.path.splitext(filename)[-1]
        loader_fn = ext_to_load_fn_map.get(ext, None)
        if loader_fn is None:
            raise ValueError("not support file type {}".format(ext))
        
        cfg = loader_fn(filename, all_unsafe)
        base_cfg_file = cfg.pop(Config._BASE_CONFIG, None)
        recursive_update_base_cfg = cfg.pop(Config._RECURSIVE_UPDATE_BASE_CONFIG, False)
        
        if base_cfg_file is None:
            return Config(cfg)
        else:
            if base_cfg_file.startswith("~"):
                base_cfg_file = os.path.expanduser(base_cfg_file)
            if not base_cfg_file.startswith("/"):
                base_cfg_file = os.path.join(os.path.dirname(filename), base_cfg_file)
            base_cfg = Config.load_file(base_cfg_file,all_unsafe=all_unsafe)
            
            def merge_a_into_b(a,b):
                for k,v in a.items():
                    if recursive_update_base_cfg:
                        if isinstance(v, dict) and k in b:
                            merge_a_into_b(v,b[k])
                        else:
                            b[k] = v
                    else:
                        b[k] = v
            merge_a_into_b(cfg,base_cfg)
            return Config(base_cfg)
        
    def __init__(self, cfg_dict=None):
        assert isinstance(cfg_dict, dict)
        new_dict = {}
        for k,v in cfg_dict.items():
            if isinstance(v, dict):
                new_dict[k] = Config(v)
            elif isinstance(v,(list, tuple)):
                v = v.__class__([Config(item) if isinstance(item, dict) else item for item in v])
                new_dict[k] = v
                
            else:
                new_dict[k] = v
        super(Config, self).__init__(new_dict)
        self.__dict__[Config._MUTABLE] = True
        
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("no attribute named {}".format(name)) 
        
    def __setattr__(self, name, value):
        # 设置属性，如果这个属性已经存在，就报错，如果不存在，就设置
        assert name not in self.__dict__, 'Invalid attribute name {}'.format(name)
        if self.__dict__[Config._MUTABLE]:
            if isinstance(value, dict):
                value = Config(value)
            elif isinstance(value, (list, tuple)):
                value = value.__class__([Config(item) if isinstance(item, dict) else item for item in value])
            self[name] = value
        else:
            raise AttributeError("Attempted to set {} to {}, but Config is immutable".format(name, value))
        
    @staticmethod
    def _recuresive_visit(obj, fn): # 递归访问
        '''这段代码的作用是通过递归方式访问给定对象及其所有子对象，并对每个对象执行特定的函数。
        这对于处理嵌套结构的对象，比如字典中包含字典、列表中包含字典等，非常有用。'''
        if isinstance(obj, Config):
            fn(obj)
        if isinstance(obj, dict):
            for value in obj.values():
                Config._recuresive_visit(value, fn)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                Config._recuresive_visit(value, fn)
    
    def set_immutable(self):
        '''设置为不可变 ,可变和不可变是指是否可以被修改'''
        def _fn(obj):
            obj.__dict__[Config._MUTABLE] = False
        self._recuresive_visit(self, _fn)
    
    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s
        r = ''
        s = []
        for k, v in sorted(self.items()):
            separator = '\n' if isinstance(v, Config) else ' '
            attr_str = '{}{}{}'.format(str(k), separator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += '\n'.join(s)
        return r
    
    @staticmethod
    def _to_str(obj):
        if isinstance(obj, Config):
            return obj._to_str()
        elif isinstance(obj, (list, tuple)):
            str_value = []
            for sub in obj:
                str_value.append(Config._to_str(sub))
            return str_value
        elif isinstance(obj, int, float, bool, str):
            return obj.__str__()
        else:
            return obj
        
    def to_str(self):
        str_config = {}
        for k, v in self.items():
            str_config[k] = Config._to_str(v)
        return str_config
    
    def __repr__(self):
        return "{}({})".format(
                                self.__class__.__name__,
                                super(Config, self).__repr__()
                                )