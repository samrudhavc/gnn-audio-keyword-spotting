import torch
import torch.nn as nn
from torch.autograd import Function


'''This code is based on the following repository:'''
'''https://github.com/Jermmy/pytorch-quantization-demo'''

def quantize_tensor(tensor: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor,
                    num_bits: int = 8,
                    signed: bool = False):
        
        '''Quantize tensor'''
        if signed:
                qmin = - 2. ** (num_bits - 1)
                qmax = 2. ** (num_bits - 1) - 1
        else:
                qmin = 0.
                qmax = 2. ** num_bits - 1.

        q_x = zero_point + (tensor / scale)
        q_x = q_x.clamp(qmin, qmax)
        q_x = q_x.round()
        
        return q_x
    
def dequantize_tensor(tensor_quant: torch.Tensor,
                      scale: torch.Tensor,
                      zero_point: torch.Tensor):
    
    '''Dequantize tensor'''
    return scale * (tensor_quant - zero_point)


class Observer(nn.Module):
    def __init__(self, 
                 num_bits:int = 8):
        super().__init__()

        self.num_bits = num_bits

        '''Initialize parameters for quantization'''
        self.register_buffer('scale', torch.tensor(0.0, requires_grad=False))
        self.register_buffer('zero_point', torch.tensor(0.0, requires_grad=False))
        self.register_buffer('min', torch.tensor(float('inf'), requires_grad=False))
        self.register_buffer('max', torch.tensor(float('-inf'), requires_grad=False))

    def update(self, tensor: torch.Tensor):
        
        '''Update parameters for quantization'''
        with torch.no_grad():
            tensor_min = torch.min(tensor).item()
            tensor_max = torch.max(tensor).item()
            self.min = torch.tensor(min(self.min.item(), tensor_min), device=tensor.device)
            self.max = torch.tensor(max(self.max.item(), tensor_max), device=tensor.device)

            if self.max > self.min:
                self.scale, self.zero_point = self.calcScaleZeroPoint()

    def quantize_tensor(self, tensor: torch.Tensor):
        
        '''Quantize tensor'''
        return quantize_tensor(tensor, self.scale, self.zero_point, self.num_bits)
    
    def dequantize_tensor(self, tensor_quant: torch.Tensor):
        
        '''Dequantize tensor'''
        return dequantize_tensor(tensor_quant, self.scale, self.zero_point)

    def calcScaleZeroPoint(self):

        '''Calculate scale and zero point for quantization'''
        qmin = 0.
        qmax = 2. ** self.num_bits - 1.

        scale = (self.max - self.min) / (qmax - qmin)
        zero_point = qmax - self.max / scale
        zero_point = zero_point.clamp(qmin, qmax).round()

        return scale, zero_point
    

class FakeQuantize(Function):
    '''Function for fake quantization.'''
    '''This function is used to calculate loss that occurs due to quantization.'''
    @staticmethod
    def forward(ctx, x, qparam):
        x = qparam.quantize_tensor(x)
        x = qparam.dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None