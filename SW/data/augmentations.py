import torch
import numpy as np

from torch_geometric.transforms import BaseTransform


class RandomShiftChannel(BaseTransform):
    def __init__(self, 
                 channels: int = 700,
                 shift: float = 5):
        
        self.channels = channels
        self.shift = shift

    # def __call__(self, data):
    def forward(self, data):
        shift = ((torch.rand(1) * 2 - 1) * self.shift).to(torch.int16)
        data.pos[:, 1] += shift
        data.pos[:, 1] = torch.clamp(data.pos[:, 1], 0, self.channels - 1)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(shift={self.shift})'
    

class RandomShiftTime(BaseTransform):
    def __init__(self, 
                 time_window: float = 1,
                 shift: float = 0.05):
        
        self.time_window = time_window
        self.shift = shift

    # def __call__(self, data):
    def forward(self, data):
        shift = ((torch.rand(1) * 2 - 1) * self.shift * self.time_window).to(torch.int16)
        data.pos[:, 0] += shift
        data.pos[:, 0] = torch.clamp(data.pos[:, 0], 0, self.time_window)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(time_window={self.time_window})'
    

class RandomSpreadChannel(BaseTransform):
    def __init__(self, 
                 channels: int = 700,
                 spread: float = 5):
        
        self.channels = channels
        self.spread = spread

    # def __call__(self, data):
    def forward(self, data):
        spread = ((torch.rand(data.pos.size(0)) * 2 - 1) * self.spread).to(torch.int16)
        data.pos[:, 1] += spread
        data.pos[:, 1] = torch.clamp(data.pos[:, 1], 0, self.channels - 1)
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(spread={self.spread})'
    

class RandomSpreadTime(BaseTransform):
    def __init__(self, 
                 time_window: float = 1,
                 spread: float = 0.05):
        
        self.time_window = time_window
        self.spread = spread
    
    # def __call__(self, data):
    def forward(self, data):
        spread = ((torch.rand(data.pos.size(0)) * 2 - 1) * self.spread * self.time_window).to(torch.int16)
        data.pos[:, 0] += spread
        data.pos[:, 0] = torch.clamp(data.pos[:, 0], 0, self.time_window)
        return data
    
    def __repr__(self):
        return f'{self.__class__.__name__}(time_window={self.time_window})'
    

class RandomRemoveNodes(BaseTransform):
    def __init__(self, 
                 remove_ratio: float = 0.05):
        
        self.remove_ratio = remove_ratio

    # def __call__(self, data):
    def forward(self, data):
        mask = torch.rand(data.pos.size(0)) < self.remove_ratio
        data.pos = data.pos[~mask]
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(remove_ratio={self.remove_ratio})'