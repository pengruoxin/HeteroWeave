"""Reference-aligned metrics and explicit legacy/context diagnostics."""
import math
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def finite(x):
    x=float(x)
    if not math.isfinite(x):raise ValueError('nonfinite score')
    return x

def weight_layers(model):
    return [(n,m) for n,m in model.named_modules() if isinstance(m,(nn.Conv2d,nn.Linear))]

class ForwardAudit:
    def __init__(self,model,n,activation_types):
        self.n=n;self.kernel_all=torch.zeros(n,n,dtype=torch.float64)
        self.kernel_relu=torch.zeros_like(self.kernel_all)
        self.counts=[];self.handles=[];self.fisher_legacy=[];self.fisher_cloned_legacy_axis=[];self.fisher_channel=[];self.fisher_param=[];self.fisher_param_ref=[]
        self.shapes=[];self.activation_counts={'all':0,'relu':0}
        for name,m in model.named_modules():
            all_act=isinstance(m,activation_types)
            relu='ReLU' in type(m).__name__
            if all_act or relu:self.handles.append(m.register_forward_hook(self.activation_hook(all_act,relu)))
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                self.handles.append(m.register_forward_hook(self.fisher_hook(name,m)))
    def activation_hook(self,all_act,relu):
        def hook(module,inputs,output):
            with torch.no_grad():
                if all_act:
                    b=output.detach().reshape(self.n,-1)>0
                    a=b.double();self.kernel_all+=(a@a.T+(1-a)@(1-a).T).cpu()
                    bits=np.packbits(b.T.contiguous().cpu().numpy(),axis=1)
                    packed=bits.view(np.dtype('V'+str(bits.shape[1]))).reshape(-1)
                    self.counts.append(int(np.unique(packed).size))
                    self.activation_counts['all']+=1
                if relu:
                    b=inputs[0].detach().reshape(self.n,-1)>0;a=b.double()
                    self.kernel_relu+=(a@a.T+(1-a)@(1-a).T).cpu()
                    self.activation_counts['relu']+=1
        return hook
    def fisher_hook(self,name,layer):
        def hook(module,inputs,output):
            if not output.requires_grad:return
            activation=output.detach().clone()
            shared_activation=output.detach()
            self.shapes.append({'module':name,'type':type(layer).__name__,'shape':list(output.shape)})
            def backward(gradient):
                p=activation*gradient.detach()
                legacy=p.sum(tuple(range(2,p.ndim))) if p.ndim>2 else p
                shared=shared_activation*gradient.detach()
                shared=shared.sum(tuple(range(2,shared.ndim))) if shared.ndim>2 else shared
                shared_value=finite(.5*shared.square().mean(0).sum())
                self.fisher_legacy.append(shared_value)
                self.fisher_param_ref.append(shared_value*layer.weight[0].numel())
                self.fisher_cloned_legacy_axis.append(finite(.5*legacy.square().mean(0).sum()))
                # Conv outputs are [B,C,...]; Linear outputs are [...,C].
                if isinstance(layer,nn.Linear) and p.ndim>2:
                    reduced=p.sum(tuple(range(1,p.ndim-1)))
                else:reduced=legacy
                channel=finite(.5*reduced.square().mean(0).sum())
                self.fisher_channel.append(channel)
                self.fisher_param.append(channel*layer.weight[0].numel())
            output.register_hook(backward)
        return hook
    def close(self):
        for h in self.handles:h.remove()
    @staticmethod
    def logdet(kernel):
        sign,val=np.linalg.slogdet(kernel.numpy())
        if sign<=0:raise ValueError('nonpositive NASWOT kernel determinant')
        return finite(val)

def gradient_bundle(model,images,labels,activation_types,collect_activations=True):
    model.zero_grad(set_to_none=True)
    types_used=activation_types if collect_activations else ()
    monitor=ForwardAudit(model,len(images),types_used)
    try:
        logits=model(images);loss=F.cross_entropy(logits,labels);loss.backward()
        layers=weight_layers(model)
        gradnorm=sum(float(m.weight.grad.norm()) for _,m in layers if m.weight.grad is not None)
        snip=sum(float((m.weight*m.weight.grad).abs().double().sum()) for _,m in layers if m.weight.grad is not None)
        all_parameters=[p for p in model.parameters() if p.grad is not None]
        result={'gradnorm_ref':finite(gradnorm),'snip_ref':finite(snip),
                'gradnorm_legacy':finite(torch.sqrt(sum(p.grad.double().square().sum() for p in all_parameters))),
                'snip_legacy':finite(sum((p*p.grad).abs().double().sum() for p in all_parameters)),
                'fisher_legacy':finite(sum(monitor.fisher_legacy)),
                'fisher_cloned_legacy_axis':finite(sum(monitor.fisher_cloned_legacy_axis)),
                'fisher_param_ref':finite(sum(monitor.fisher_param_ref)),
                'fisher_channel':finite(sum(monitor.fisher_channel)),
                'fisher_param':finite(sum(monitor.fisher_param)),
                'loss':finite(loss),'weight_layers':len(layers),
                'missing_weight_gradients':[n for n,m in layers if m.weight.grad is None],
                'linear_token_layers':sum(s['type'].endswith('Linear') and len(s['shape'])>2 for s in monitor.shapes)}
        if monitor.activation_counts['relu']:
            result['naswot_relu']=monitor.logdet(monitor.kernel_relu)
        if collect_activations:
            result.update(clas=finite(sum(math.sqrt(n) for n in monitor.counts)),
                          no_sqrt=finite(sum(monitor.counts)),
                          naswot_all=monitor.logdet(monitor.kernel_all),
                          activation_counts=monitor.activation_counts)
        return result
    finally:monitor.close();model.zero_grad(set_to_none=True)

def synflow_bundle(model,images,absolute_buffers=False,no_bn=False):
    """One factor at a time; restore every tensor and patched forward."""
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    dtype=next(model.parameters()).dtype
    patched=[]
    model.zero_grad(set_to_none=True)
    try:
        model.double()
        with torch.no_grad():
            targets=model.state_dict().values() if absolute_buffers else model.parameters()
            for p in targets:p.abs_()
        if no_bn:
            for m in model.modules():
                if isinstance(m,(nn.BatchNorm1d,nn.BatchNorm2d)):
                    patched.append((m,'forward' in m.__dict__,m.__dict__.get('forward')))
                    m.forward=types.MethodType(lambda self,x:x,m)
        model(torch.ones_like(images[:1],dtype=torch.float64)).sum().backward()
        all_score=sum(float((p*p.grad).abs().sum()) for p in model.parameters() if p.grad is not None)
        weight_score=sum(float((m.weight*m.weight.grad).abs().sum()) for _,m in weight_layers(model) if m.weight.grad is not None)
        assert all_score>0 and weight_score>0
        return {'all_parameters_log':finite(math.log(all_score)),
                'conv_linear_weights_log':finite(math.log(weight_score)),
                'bn_bypassed':len(patched)}
    finally:
        for m,existed,old in patched:
            if existed:m.forward=old
            else:del m.forward
        model.to(dtype=dtype);model.load_state_dict(state,strict=True)
        model.zero_grad(set_to_none=True)

def zico_bundle(model,images,labels,official_formula):
    gradients={};missing=set()
    for i,left in enumerate(range(0,len(images),8)):
        model.zero_grad(set_to_none=True)
        loss=F.cross_entropy(model(images[left:left+8]),labels[left:left+8]);loss.backward()
        for name,m in weight_layers(model):
            if m.weight.grad is None:missing.add(name);continue
            gradients.setdefault(name,[]).append(m.weight.grad.detach().reshape(-1).cpu().numpy().copy())
    model.zero_grad(set_to_none=True)
    return finite(official_formula(gradients)),sorted(missing)
