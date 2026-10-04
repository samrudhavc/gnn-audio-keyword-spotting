import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data
from typing import Optional, Union

from models.networks.layers.quantisation.observer import Observer, FakeQuantize, quantize_tensor, dequantize_tensor

Pooling = {
    'mean': global_mean_pool,
    'add': global_add_pool,
    'max': global_max_pool
}

class MyGlobalPooling(nn.Module):
    def __init__(
        self,
        aggregator: str = 'mean',
        num_bits: int = 8
    ):
        super().__init__()

        self.aggregator = Pooling[aggregator]
        self.num_bits = num_bits

        self.register_buffer('calib_mode', torch.tensor(False, requires_grad=False))
        self.register_buffer('quantize_mode', torch.tensor(False, requires_grad=False))

    def forward(
        self,
        data: Data,
        observer: Observer
    ) -> Tensor:
        
        if self.calib_mode and not self.quantize_mode:
            out = self.aggregator(data.x, data.batch)
            out = FakeQuantize.apply(out, observer)
        elif self.quantize_mode:
            out = self.aggregator(data.x, data.batch)
            out = torch.clamp(out, 0, 2**self.num_bits-1)
            out = out.round()
            out = observer.dequantize_tensor(out)
        elif not self.calib_mode and not self.quantize_mode:
            out = self.aggregator(data.x, data.batch)
        else:
            raise ValueError('Invalid mode')
        
        return out
    
    def calibrate(self):
        self.calib_mode.fill_(True)

    def quantize(self):
        self.quantize_mode.fill_(True)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__} (num_bits={self.num_bits})')