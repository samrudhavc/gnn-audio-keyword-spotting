import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sequential, Linear, BatchNorm1d
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.typing import (
    Adj,
    OptTensor,
    PairOptTensor,
    PairTensor,
    SparseTensor,
    torch_sparse,
)
from typing import Optional, Union

from models.networks.layers.quantisation.observer import Observer, FakeQuantize, quantize_tensor, dequantize_tensor

class MyPointNetConv(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = False,
        num_bits: int = 8,
        first_layer: bool = False,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'max')
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_bits = num_bits
        self.bias = bias
        self.first_layer = first_layer

        # Number of bits for quantization scales
        self.num_bits_obs = 32 

        # Define layers
        self.linear = Linear(input_dim, output_dim, bias=bias)
        self.norm = BatchNorm1d(output_dim)
        self.mlp = Sequential(self.linear, self.norm)

        self.use_relu = True
        self.local_nn = self.mlp
        self.global_nn = None  # Currently not used
        self.add_self_loops = True

        self.reset_parameters()

        # Modes for calibration and quantization
        self.register_buffer('calib_mode', torch.tensor(False, requires_grad=False))
        self.register_buffer('quantize_mode', torch.tensor(False, requires_grad=False))

        # Initialize quantization observers
        self.observer_input = Observer(num_bits=num_bits)
        self.observer_weight = Observer(num_bits=num_bits)
        self.observer_output = Observer(num_bits=8)

        # Register buffers for quantization parameters
        self.register_buffer('m', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_in', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_w', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_out', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('qscale_m', torch.tensor(1.0, requires_grad=False))
        self.register_buffer('num_bits_model', torch.tensor(num_bits, requires_grad=False))
        self.register_buffer('num_bits_scale', torch.tensor(self.num_bits_obs, requires_grad=False))

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.local_nn)
        reset(self.global_nn)

    def forward(
        self,
        data: Data,
    ) -> Tensor:
        
        '''
            Standard forward method of a PointNetConv layer
        '''

        x = data.x
        pos = data.pos
        edge_index = data.edge_index

        if not isinstance(x, tuple):
            x = (x, x)

        if isinstance(pos, Tensor):
            pos = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(
                    edge_index, num_nodes=min(pos[0].size(0), pos[1].size(0)))
            elif isinstance(edge_index, SparseTensor):
                edge_index = torch_sparse.set_diag(edge_index)

        # propagate_type: (x: PairOptTensor, pos: PairTensor)
        out = self.propagate(edge_index, x=x, pos=pos)

        if self.global_nn is not None:
            out = self.global_nn(out)
        
        # Apply activation function
        if self.use_relu:
            if not self.quantize_mode:
                out = F.relu(out)
            else:
                # In quantize mode, simulate quantized ReLU
                out[out < self.observer_output.zero_point] = self.observer_output.zero_point

        return out

    def message(self, x_i: Optional[Tensor], x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        '''
            Custom message function for PointNetConv.
            We select the message function based on the current mode (calibration, quantize, or float)
        '''
        if self.calib_mode and not self.quantize_mode:
            return self.message_calib(x_i, x_j, pos_i, pos_j)
        elif self.quantize_mode:
            return self.message_quant(x_i, x_j, pos_i, pos_j)
        elif not self.calib_mode and not self.quantize_mode:
            return self.message_float(x_i, x_j, pos_i, pos_j)
        else:
            raise ValueError('Invalid mode')

    def message_float(self, x_i: Optional[Tensor], x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i

        msg[:, 0] *= -50 # 1/radius_time [in seconds]
        msg[:, 1] += 1/7 # radius_channel / num_channels
        msg[:, 1] *= 7/2 # num_channels / (2*radius_channel)

        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)
        if self.local_nn is not None:
            msg = self.local_nn(msg)
        return msg

    def message_calib(self, x_i: Optional[Tensor], x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i

        msg[:, 0] *= -50 # 1/radius_time [in seconds]
        msg[:, 1] += 1/7 # radius_channel / num_channels
        msg[:, 1] *= 7/2 # num_channels / (2*radius_channel)

        if x_j is not None:
            msg = torch.cat([x_j, msg], dim=1)

        # Update input observer
        if self.training:
            self.observer_input.update(msg)
        msg = FakeQuantize.apply(msg, self.observer_input)

        # Simulate batch normalization during calibration
        if self.training:
            y = torch.nn.functional.linear(msg, self.linear.weight, self.linear.bias)
            _ = self.norm(y)
        
        # Merge batch normalization parameters with linear weights
        mean = self.norm.running_mean
        var = self.norm.running_var
        std = torch.sqrt(var + self.norm.eps)
        weight, bias = self.merge_norm(mean, std)

        # Update weight observer
        if self.training:
            self.observer_weight.update(weight)

        # Apply quantized weights
        if self.local_nn is not None:
            weight_q = FakeQuantize.apply(weight, self.observer_weight)
            msg = F.linear(msg, weight_q, bias)

        # Update output observer
        if self.training:
            self.observer_output.update(msg)
            self.observer_output.update(pos_j-pos_i) # Update observer for pos_j-pos_i to avoid quantization error

        msg = FakeQuantize.apply(msg, self.observer_output)
        return msg
    
    def message_quant(self, x_i: Optional[Tensor], x_j: Optional[Tensor], pos_i: Tensor, pos_j: Tensor) -> Tensor:
        msg = pos_j - pos_i

        msg[:, 0] *= -50 # 1/radius_time [in seconds]
        msg[:, 1] += 1/7 # radius_channel / num_channels
        msg[:, 1] *= 7/2 # num_channels / (2*radius_channel)

        '''
            Quantize input message, if first layer we need to quantize both x_j and pos differences
            If not first layer, we quantize only the pos differences and concatenate with x_j
        '''
        
        if self.first_layer:
            msg = torch.cat([x_j, msg], dim=1)
            msg = self.observer_input.quantize_tensor(msg)

        else:
            msg = self.observer_input.quantize_tensor(msg)
            msg = torch.cat([x_j, msg], dim=1)
        msg = msg - self.observer_input.zero_point

        # Apply quantized linear layer
        msg = self.qlinear(msg)

        # Requantize output message
        msg = msg * self.m
        msg = msg.round() 
        msg = msg + self.observer_output.zero_point   

        # Clamp output message
        msg = torch.clamp(msg, 0, 2**8-1)
        msg = msg.round()

        deq = self.observer_output.dequantize_tensor(msg)
        return msg

    def merge_norm(self,
                   mean: torch.Tensor,
                   std: torch.Tensor):
        
        '''
            Merge batch normalization parameters with linear weights.
        '''

        if self.norm.affine:
            gamma = self.norm.weight
            beta = self.norm.bias
        else:
            gamma = torch.ones_like(std)
            beta = torch.zeros_like(std)

        W = self.linear.weight

        if self.bias:
            b = self.linear.bias
        else:
            b = torch.zeros(self.output_dim, device=W.device)
        
        W_new = (gamma / std).unsqueeze(1) * W
        b_new = (gamma / std) * (b - mean) + beta
        
        return W_new, b_new
    
    def calibrate(self):
        self.calib_mode.fill_(True)

    def quantize(self,
               observer_input: Observer = None,
               observer_output: Observer = None):
        '''
            Quantize model - quantize weights/bias and calculate scales
        '''
        self.quantize_mode.fill_(True)

        if observer_input is not None:
            self.observer_input = observer_input
        if observer_output is not None:
            self.observer_output = observer_output

        # Quantize scales for input, weight, and output
        self.qscale_in = (2 ** self.num_bits_obs) * self.observer_input.scale
        self.qscale_in = self.qscale_in.round()
        self.observer_input.scale = self.qscale_in / (2 ** self.num_bits_obs)

        self.qscale_w = (2 ** self.num_bits_obs) * self.observer_weight.scale
        self.qscale_w = self.qscale_w.round()
        self.observer_weight.scale = self.qscale_w / (2 ** self.num_bits_obs)

        self.qscale_out = (2 ** self.num_bits_obs) * self.observer_output.scale
        self.qscale_out = self.qscale_out.round()
        self.observer_output.scale = self.qscale_out / (2 ** self.num_bits_obs)

        # Compute scaling factor m
        self.qscale_m = (self.observer_weight.scale * self.observer_input.scale) / self.observer_output.scale
        self.qscale_m = self.qscale_m * (2 ** self.num_bits_obs)
        self.qscale_m = self.qscale_m.round()
        self.m = self.qscale_m / (2 ** self.num_bits_obs)

        # Merge batch normalization parameters
        std = torch.sqrt(self.norm.running_var + self.norm.eps)
        weight, bias = self.merge_norm(self.norm.running_mean, std)

        with torch.no_grad():
            # Initialize quantized linear layer
            self.qlinear = Linear(self.input_dim, self.output_dim, bias=True)

            # Quantize weights
            quantized_weight = self.observer_weight.quantize_tensor(weight)
            quantized_weight = quantized_weight - self.observer_weight.zero_point
            self.qlinear.weight.copy_(quantized_weight)

            # Quantize biases
            quantized_bias = quantize_tensor(
                bias,
                scale=self.observer_weight.scale * self.observer_input.scale,
                zero_point=0,
                num_bits=32,
                signed=True,
            )
            self.qlinear.bias.copy_(quantized_bias)

            self.qlinear.to(self.linear.weight.device)
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(local_nn={self.local_nn}, '
                f'global_nn={self.global_nn}), num_bits={self.num_bits}')