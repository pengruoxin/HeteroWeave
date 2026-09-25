

import torch
from .jacov import jacov
from .grad_norm import grad_norm
from .naswot import naswot
from .synflow import synflow
from .snip import snip
from .fisher import fisher
from .ntk import nasi
from .zico import (
    zico, collect_zico_batch, calculate_zico, calculate_normalized_zico)
from tqdm import tqdm
from torch import nn

import numpy as np

ZERO_PROXY = {
    'jacov': jacov,
    'grad_norm': grad_norm,
    'naswot': naswot,
    'synflow': synflow,
    'snip': snip,
    'fisher': fisher,
    'ntk': nasi,
    'nasi': nasi,
    'zico': zico
}


class ZeroNas:
    def __init__(self, dataloader,
                 indicator,
                 criterion=nn.CrossEntropyLoss(),
                 num_batch=1):
        assert isinstance(indicator, str) or isinstance(indicator, list)
        if isinstance(indicator, str):

            self.ntk = True if indicator in ['ntk', 'nasi'] else False

            self.indicator = {indicator: ZERO_PROXY[indicator]}
        elif isinstance(indicator, list):

            self.ntk = True if indicator[0] in ['ntk', 'nasi'] else False
            self.indicator = {ind: ZERO_PROXY[ind] for ind in indicator}

        self.dataloader = dataloader

        self.criterion = criterion
        self.num_batch = num_batch

    def get_score(self, model):
        model = model.cuda()
        model.train()
        scores = dict()
        zico_grad_dict = {}
        zico_step = 0
        has_zico = 'zico' in self.indicator
        for i, data in enumerate(self.dataloader):
            x = data['img'].cuda()
            y = data['gt_label'].cuda()
            for k, score_func in self.indicator.items():
                if k == 'zico':
                    continue
                if k in scores.keys():
                    scores[k].append(score_func(
                        model, x, y, self.criterion))
                else:
                    scores[k] = [score_func(model, x, y, self.criterion)]

            if has_zico:
                zico_grad_dict = collect_zico_batch(
                    model, x, y, self.criterion, zico_grad_dict, zico_step)
                zico_step += 1

            if i+1 == self.num_batch:
                break

        for k, v in scores.items():
            # Get the mean value
            scores[k] = sum(v)/len(v)
        if has_zico:
            scores['zico'] = calculate_zico(zico_grad_dict)
            normalized_zico = calculate_normalized_zico(zico_grad_dict)
            scores['normalized_zico_mean'] = normalized_zico['mean']
            scores['normalized_zico_top'] = normalized_zico['top']
            scores['normalized_zico_clip'] = normalized_zico['clip']
            scores['normalized_zico_mean_dispersion'] = normalized_zico[
                'mean_dispersion']
            scores['normalized_zico_top_dispersion'] = normalized_zico[
                'top_dispersion']
            scores['normalized_zico_clip_dispersion'] = normalized_zico[
                'clip_dispersion']
            scores['normalized_zico_mean_stable'] = normalized_zico['mean_stable']
            scores['normalized_zico_top_stable'] = normalized_zico['top_stable']
            scores['normalized_zico_clip_stable'] = normalized_zico['clip_stable']
            scores['normalized_zico_balanced'] = normalized_zico['balanced']

        return scores
