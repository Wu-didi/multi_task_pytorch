from __future__ import absolute_import,print_function # Python 2/3 compatibility
from collections import OrderedDict
import torch.nn as nn
import copy
from torchmetrics import MetricCollection
import inspect



def _as_list(obj):
    if isinstance(obj, (list or tuple)):
        return obj
    else:
        return [obj]
    

# 存在问题，需要修改 bug
def Merge_model_export(state_dict, model_list):
    '''
    state_dict: the state_dict of the model,its weight
    model_list: the model list , model structure
    '''
    for idx, model in enumerate(model_list):
        for name_i in model.named_childern(): # name_i is a tuple, e.g. ("head",moduleList)
            if "head" in name_i[0]:
                name_module = name_i[1]._modules # name_module is a OrderedDict
                assert isinstance(name_module, nn.ModuleList), "type of name_module should be nn.ModuleList vs. {}".format(type(name_module))

                for k,v in name_module.items():
                    if k == '0':
                        d2 = OrderedDict([(str(idx),v)])
                        name_i[1]._modules = d2
                    else:
                        raise ValueError("the key of name_module should be 1 vs. {}".format(len(name_module)))
    model_export = copy.copy(model_list[0])
    d3 = dict()
    for idx, model in enumerate(model_list):
        for name_i in model.named_childern():
            if "head" in name_i[0]:
                name_module = name_i[1]._modules
                for k,v in name_module.items():
                    d3[k] = v
    d3 = OrderedDict(d3)
    
    for name_i in model_export.named_childern():
        if "head" in name_i[0]:
            name_i[1]._modules = d3
    model_export.load_state_dict(state_dict=state_dict, strict=True)
    return model_export
