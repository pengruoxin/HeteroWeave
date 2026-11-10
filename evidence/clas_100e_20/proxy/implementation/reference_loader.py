"""Load reviewed reference function definitions without repository import effects."""
import ast
from pathlib import Path
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def load_functions(path, names, namespace=None):
    ns=namespace if namespace is not None else {}
    ns.update(torch=torch,nn=nn,F=F,np=np,types=types)
    nodes=[]
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            if isinstance(node,ast.FunctionDef):node.decorator_list=[d for d in node.decorator_list if not isinstance(d,ast.Call) or not isinstance(d.func,ast.Name) or d.func.id!='measure']
            nodes.append(node)
    assert {n.name for n in nodes}==set(names),(path,names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns

def reference_namespace(root):
    root=Path(root);ns={}
    load_functions(root/'p_utils.py',['get_layer_metric_array','reshape_elements'],ns)
    load_functions(root/'grad_norm.py',['get_grad_norm_arr'],ns)
    load_functions(root/'snip.py',['snip_forward_conv2d','snip_forward_linear','compute_snip_per_weight'],ns)
    load_functions(root/'fisher.py',['fisher_forward_conv2d','fisher_forward_linear','compute_fisher_per_weight'],ns)
    load_functions(root/'jacob_cov.py',['get_batch_jacobian','eval_score','compute_jacob_cov'],ns)
    load_functions(root/'synflow.py',['compute_synflow_per_weight'],ns)
    load_functions(root/'zico_official.py',['getgrad','caculate_zico'],ns)
    return ns

def sum_arrays(values):
    return float(sum(v.detach().double().sum().item() for v in values))
