from __future__ import absolute_import, print_function # Python 2/3 compatibility
import os
import sys
from importlib import import_module
import json
import yaml


def _check_path_exist(path, local_only=False):
    if local_only:
        fn = os.path.exists
    else:
        raise ValueError("local only should set true")
    assert fn(path), "path {} not exist".format(path)
    
def load_pyfile(filename, allow_unsafe=False):
    _check_path_exist(filename, local_only=True)
    module_name = os.path.basename(filename)[:-3]
    config_dir = os.path.dirname(filename)
    sys.path.insert(0, config_dir)
    mod = import_module(module_name)
    sys.path.pop(0)
    cfg = {name: value for name, value in mod.__dict__.items()
                if not name.startswith('__')}
    sys.modules.pop(module_name)
    return cfg

def load_json(filename, all_unsafe=False):
    _check_path_exist(filename)
    with open(filename, 'r') as f:
        cfg = json.load(f)
    return cfg

def load_yaml(filename, all_unsafe=False):
    _check_path_exist(filename)
    with open(filename, 'r') as f:
        if all_unsafe:
            cfg = yaml.unsafe_load(f)
        else:
            cfg = yaml.safe_load(f)
    return cfg

ext_to_load_fn_map = {
    '.py': load_pyfile,
    '.json': load_json,
    '.yaml': load_yaml,
    '.yml': load_yaml,
}