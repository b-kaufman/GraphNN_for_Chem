import torch
from layers import ScratchNNConv, DuvenaudMP, global_sum_pool, ElistNNConv, SparseNodeConv, SparseNodeEdgeConv
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torch_geometric.nn import graclus, max_pool, global_mean_pool
import torch.nn.functional as F
import numpy as np


class GraphSparseNodeOnly(torch.nn.Module):

    def __init__(self, input_features, num_conv_layers, conv_output_dims, num_fc_layers, fc_output_dims, debug=False):
        super(GraphSparseNodeOnly, self).__init__()

        assert num_conv_layers == len(conv_output_dims)
        assert num_fc_layers == len(fc_output_dims)
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.input_features = input_features
        self.conv_layers = [SparseNodeConv(self.input_features, conv_output_dims[idx]) if idx == 0
                            else SparseNodeConv(conv_output_dims[idx-1],
                                                conv_output_dims[idx]) for idx in range(num_conv_layers)]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.fc_layers = [torch.nn.Linear(conv_output_dims[-1], fc_output_dims[idx]) if idx == 0
                          else torch.nn.Linear(fc_output_dims[idx-1],
                                               fc_output_dims[idx]) for idx in range(num_fc_layers)]
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        if debug:
            self.debug()

    def forward(self, data):
        for ii in range(self.num_conv_layers):
            data.node_attr = self.conv_layers[ii](data.node_attr, data.node_mask).clamp(0)
        output = global_sum_pool(data.node_attr, data.batching)
        for ii in range(self.num_fc_layers):
            output = self.fc_layers[ii](output)
        return F.softmax(output, dim=1)

    def debug(self):
        return

class EdgeConditionedConv(torch.nn.Module):

    def __init__(self, node_input_features, edge_input_features, num_conv_layers, conv_output_dims, num_edge_net_layers, edge_net_output_dims, num_fc_layers, fc_output_dims, mode='classification', debug=False):
        super(EdgeConditionedConv, self).__init__()
        self.mode = mode
        self.node_input_features = node_input_features
        self.edge_input_features = edge_input_features
        self.num_conv_layers = num_conv_layers
        self.num_edge_net_layers = num_edge_net_layers
        self.num_fc_layers = num_fc_layers
        self.conv_layers = [SparseNodeEdgeConv(self.node_input_features, edge_input_features, conv_output_dims[idx], edge_net_output_dims) if idx == 0
                            else SparseNodeEdgeConv(conv_output_dims[idx - 1], edge_input_features,
                                                conv_output_dims[idx], edge_net_output_dims) for idx in range(num_conv_layers)]
        self.conv_layers = torch.nn.ModuleList(self.conv_layers)
        self.fc_layers = [torch.nn.Linear(conv_output_dims[-1], fc_output_dims[idx]) if idx == 0
                          else torch.nn.Linear(fc_output_dims[idx - 1],
                                               fc_output_dims[idx]) for idx in range(num_fc_layers)]
        self.fc_layers = torch.nn.ModuleList(self.fc_layers)
        if debug:
            self.debug()

    def forward(self, data):
        for ii in range(self.num_conv_layers):
            data.node_attr = self.conv_layers[ii](data.node_attr, data.edge_attr, data.node_mask, data.edge_mask).clamp(min=0)
        output = global_sum_pool(data.node_attr, data.batching)
        for ii in range(self.num_fc_layers-1):
            output = self.fc_layers[ii](output).clamp(min=0)
        output = self.fc_layers[-1](output)
        if self.mode == 'classification':
            return F.softmax(output, dim=1)
        if self.mode == 'regression':
            assert output.size(1) == 1
            return output

    def debug(self):
        return

class DuvenaudClass(torch.nn.Module):
    def __init__(self, out_dim, input_features,
                 layers_num, model_dim, dropout
                 ):
        super(DuvenaudClass, self).__init__()
        self.layers_num = layers_num
        self.conv_layers = [DuvenaudMP(input_features, model_dim, dropout=dropout)] + \
                           [DuvenaudMP(model_dim, model_dim) for _ in range(layers_num - 1)]

        self.conv_layers = torch.nn.ModuleList(self.conv_layers)

        self.fc1 = torch.nn.Linear(model_dim, out_dim)

    def forward(self, data):
        for i in range(self.layers_num):
            data.x = self.conv_layers[i](data.x, data.edge_index, data.edge_attr)
        data.x = global_mean_pool(data.x, data.batch)
        x = self.fc1(data.x)

        return F.softmax(x, dim=1)
