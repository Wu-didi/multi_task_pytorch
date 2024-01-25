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
    
    def _do_register(self, name, obj)