from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low Rank Adaptation"""

    def __init__(self, original, rank=4) -> None:
        super().__init__()

        self.original = original
        for param in self.original.parameters():
            param.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x))


def inject_lora(module, target_layer_name, rank=4):
    """Replace a linear layer inside module with a LoRA-wrapped version."""
    parts = target_layer_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)

    layer_name = parts[-1]
    original_layer = getattr(parent, layer_name)

    if not isinstance(original_layer, nn.Linear):
        raise TypeError(
            f"Expected nn.Linear at '{target_layer_name}', "
            f"got {type(original_layer).__name__}"
        )

    lora_layer = LoRALinear(original_layer, rank=rank)
    setattr(parent, layer_name, lora_layer)
