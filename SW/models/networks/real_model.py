import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, PointNetConv, GATConv, GATv2Conv, GINConv, GAT, GIN, global_mean_pool, \
    global_add_pool, global_max_pool, SplineConv, BatchNorm, PairNorm
from torch.nn import Module, ModuleList, Linear, Dropout, Sequential
from torch_geometric.data import Data


from models.networks.layers.my_pointnet import MyPointNetConv
from models.networks.layers.my_pooling import MyGlobalPooling


class GCN(Module):
    def __init__(self, 
                 config):
        super(GCN, self).__init__()
        
        self.config = config

        conv_ch = config.model.conv_channels
        conv_bits = config.model.conv_bits

        linear_ch = config.model.linear_channels
        num_classes = config.model.num_classes

        input_dim = 2 if config.graph.features else 0
        
        self.conv1 = MyPointNetConv(input_dim+2, conv_ch[0], bias=False, num_bits=conv_bits[0], first_layer=True)
        self.conv2 = MyPointNetConv(conv_ch[0]+2, conv_ch[1], bias=False, num_bits=conv_bits[1])
        self.conv3 = MyPointNetConv(conv_ch[1]+2, conv_ch[2], bias=False, num_bits=conv_bits[2])
        self.conv4 = MyPointNetConv(conv_ch[2]+2, conv_ch[3], bias=False, num_bits=conv_bits[3])
        
        self.pooling = MyGlobalPooling(config.model.global_pooling, num_bits=conv_bits[3])

        self.rnn = torch.nn.LSTM(input_size=conv_ch[3], hidden_size=conv_ch[3], num_layers=1, batch_first=True)

        # self.fc1 = Linear(4 * conv_ch[3], linear_ch)
        self.fc1 = Linear(conv_ch[3], linear_ch)
        self.fc2 = Linear(linear_ch, num_classes)


    def forward(self, data):
        outputs = []

        data.x = self.conv1(data)
        data.x = self.conv2(data)
        data.x = self.conv3(data)
        data.x = self.conv4(data)

        # for i in range(7):
        #     start = i * 1 / 7
        #     end = (i + 1) * 1 / 7
        #     mask = (data.pos[:, 0] >= start) & (data.pos[:, 0] < end)

        #     new_data = Data(pos=data.pos[mask],
        #                     x=data.x[mask],
        #                     batch=data.batch[mask])

        #     x = self.pooling(new_data, self.conv4.observer_output)
        #     out = self.rnn(x.unsqueeze(1))[0].squeeze(1)  # (batch_size, seq_len, hidden_size)

        out = self.pooling(data, self.conv4.observer_output)

        x = self.fc1(out)
        x = torch.relu(x)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x, outputs
    
    def calibrate(self):
        self.conv1.calibrate()
        self.conv2.calibrate()
        self.conv3.calibrate()
        self.conv4.calibrate()
        self.pooling.calibrate()

    def quantize(self):
        self.conv1.quantize()
        self.conv2.quantize(observer_input=self.conv1.observer_output)
        self.conv3.quantize(observer_input=self.conv2.observer_output)
        self.conv4.quantize(observer_input=self.conv3.observer_output)
        self.pooling.quantize()