"""Frozen residual fusion with per-token RMS amplitude alignment."""

import torch
from torch import nn


class MultiRMSLayer(nn.Module):
    def __init__(self, clip_layer, donors, gates):
        super().__init__()
        if len(donors) != len(gates) or not donors:
            raise ValueError('Provide one fixed gate per donor')
        self.clip_layer = clip_layer
        self.donors = nn.ModuleList(donors)
        self.register_buffer('gate_values', torch.tensor(gates, dtype=torch.float32))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden_states, attention_mask, causal_attention_mask,
                output_attentions=False):
        outputs = self.clip_layer(
            hidden_states, attention_mask, causal_attention_mask,
            output_attentions=output_attentions)
        if bool((self.gate_values == 0).all()):
            return outputs
        target = (outputs[0] - hidden_states).float().square().mean(
            -1, keepdim=True).sqrt()
        updates = []
        for donor, gate in zip(self.donors, self.gate_values):
            residual = donor(hidden_states) - hidden_states
            amplitude = residual.float().square().mean(
                -1, keepdim=True).sqrt().clamp_min(1e-6)
            aligned = (residual.float() * (target / amplitude)).to(residual.dtype)
            updates.append(gate.to(residual.dtype) * aligned)
        return (outputs[0] + torch.stack(updates).sum(0),) + outputs[1:]
