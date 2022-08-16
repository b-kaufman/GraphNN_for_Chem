import torch
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
import numpy as np
import functools
import torch_sparse
import math


class ScratchNNConv(Module):
    r"""
    kernel GCN layer from
    "Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    """

    def __init__(self, in_channels, edge_channels, out_channels, bias=True):
        super(ScratchNNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.node_weight = Parameter(torch.zeros([in_channels, out_channels], dtype=torch.float32))
        self.edge_lay_1 = Parameter(torch.zeros([edge_channels, out_channels - 1], dtype=torch.float32))
        self.edge_lay_2 = Parameter(torch.zeros([out_channels - 1, out_channels], dtype=torch.float32))
        self.root = Parameter(torch.zeros([in_channels, out_channels], dtype=torch.float32))
        if bias:
            self.bias = Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        node_stdv = 1. / math.sqrt(self.node_weight.size(1))
        self.node_weight.data.uniform_(-node_stdv, node_stdv)
        self.root.data.uniform_(-node_stdv, node_stdv)
        e1_stdv = 1. / math.sqrt(self.edge_lay_1.size(1))
        e2_stdv = 1. / math.sqrt(self.edge_lay_2.size(1))
        self.edge_lay_1.data.uniform_(-e1_stdv, e1_stdv)
        self.edge_lay_2.data.uniform_(-e2_stdv, e2_stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-node_stdv, node_stdv)

    def forward(self, node_mat, adj, edge_adj):
        potential_node_msgs = torch.mm(node_mat, self.node_weight)
        sum_filter_node_msgs = torch.mm(adj, potential_node_msgs)
        e_mlp_lay_1_out = torch.einsum('ijk,ib->bjk', edge_adj, self.edge_lay_1).clamp(min=0)
        e_mlp_lay_2_out = torch.einsum('ijk,ib->bjk', e_mlp_lay_1_out, self.edge_lay_2)
        e_compress = torch.einsum('ijk,jk->ji', e_mlp_lay_2_out, adj)
        root_info = torch.mm(node_mat, self.root)
        output = sum_filter_node_msgs + e_compress + root_info
        if self.bias is not None:
            return output + self.bias
        else:
            return output


'''
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
'''


class ElistNNConv(Module):
    r"""
    Another implementation of kernel GCN layer from
    "Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    """

    def __init__(self, in_channels, edge_channels, out_channels, bias=True):
        super(ElistNNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.node_weight = Parameter(torch.zeros([in_channels, out_channels], dtype=torch.float32))
        self.edge_lay_1 = Parameter(torch.zeros([edge_channels, out_channels], dtype=torch.float32))
        #self.edge_lay_2 = Parameter(torch.zeros([out_channels - 1, out_channels], dtype=torch.float32))
        self.root = Parameter(torch.zeros([in_channels, out_channels], dtype=torch.float32))
        if bias:
            self.bias = Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        node_stdv = 1. / math.sqrt(self.node_weight.size(1))
        self.node_weight.data.uniform_(-node_stdv, node_stdv)
        self.root.data.uniform_(-node_stdv, node_stdv)
        e1_stdv = 1. / math.sqrt(self.edge_lay_1.size(1))
        #e2_stdv = 1. / math.sqrt(self.edge_lay_2.size(1))
        self.edge_lay_1.data.uniform_(-e1_stdv, e1_stdv)
        #self.edge_lay_2.data.uniform_(-e2_stdv, e2_stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-node_stdv, node_stdv)

    def forward(self, node_mat, edge_mat, e_list, device='cpu'):

        potential_node_msgs = torch.mm(node_mat, self.node_weight)
        sum_filter_node_msgs = torch_sparse.spmm(e_list, torch.ones(e_list.size()[1]).to(device),
                                                 node_mat.size()[0], potential_node_msgs)
        e_mlp_lay_1_out = torch.mm(edge_mat, self.edge_lay_1).clamp(min=0)
        emask_idx = torch.stack([e_list[1],torch.arange(0,e_list.size()[1]).to(device)],dim=0)
        e_compress = torch_sparse.spmm(emask_idx, torch.ones(e_list.size()[1]).to(device), node_mat.size()[0], e_mlp_lay_1_out)
        root_info = torch.mm(node_mat, self.root)
        output = sum_filter_node_msgs + e_compress + root_info
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SparseNodeEdgeConv(Module):
    r"""
    yet another kernel GCN layer from
    "Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper. Uses sparse encodings for more efficient storage
    """

    def __init__(self, node_in_channels, edge_in_channels, out_channels, edge_net_output_dims, bias=True):
        super(SparseNodeEdgeConv, self).__init__()
        self.node_in_channels = node_in_channels
        self.edge_in_channels = edge_in_channels
        self.edge_net_output_dims = edge_net_output_dims
        self.out_channels = out_channels
        #assert self.out_channels == self.edge_net_output_dims[-1]
        self.root = Parameter(torch.zeros([self.node_in_channels, self.out_channels], dtype=torch.float32))
        self.edge_net = [Parameter(torch.zeros([self.edge_in_channels, self.edge_net_output_dims[idx]],
                                   dtype=torch.float32)) if idx == 0
                         else Parameter(torch.zeros([self.edge_net_output_dims[idx-1], self.edge_net_output_dims[idx]],
                                                    dtype=torch.float32)) for idx in range(len(self.edge_net_output_dims))]
        self.edge_net = torch.nn.ParameterList(self.edge_net)
        self.edge_net_out_weight = Parameter(torch.zeros([self.node_in_channels, self.edge_net_output_dims[-1],
                                                          self.out_channels], dtype=torch.float32))
        if bias:
            self.bias = Parameter(torch.zeros(self.out_channels, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        node_stdv = 1. / math.sqrt(self.root.size(1))
        self.root.data.uniform_(-node_stdv, node_stdv)
        for layer in self.edge_net:
            e_stdv = 1. / math.sqrt(layer.size(1))
            layer.data.uniform_(-e_stdv, e_stdv)
        filter_stdv = 1. / math.sqrt(self.edge_net_out_weight.size(0)*self.edge_net_out_weight.size(2))
        self.edge_net_out_weight.data.uniform_(-filter_stdv,filter_stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-node_stdv, node_stdv)

    def forward(self, node_attr, edge_attr, node_mask, edge_mask):
        num_edges = edge_attr.size(0)
        root_info = torch.mm(node_attr, self.root)
        for layer in self.edge_net:
            edge_attr = torch.mm(edge_attr, layer).clamp(min=0)
        assert edge_attr.size() == torch.Size([num_edges, self.edge_net_output_dims[-1]])
        edge_out = torch.matmul(edge_attr, self.edge_net_out_weight).permute(1,0,2)
        assert edge_out.size() == torch.Size([num_edges, self.node_in_channels, self.out_channels])
        edge_out = torch.matmul(node_attr, edge_out)
        assert edge_out.size() == torch.Size([num_edges, node_attr.size(0), self.out_channels])
        #print(edge_mask.size(),edge_out.size())
        #e_mask - 12x26 e_mask.t - 26x12x1
        #print(torch.t(edge_mask).unsqueeze(2).size(),edge_out.size())
        edge_out = torch.mul(torch.t(edge_mask).unsqueeze(2), edge_out)
        edge_out = torch.sum(edge_out, dim=0)
        assert edge_out.size() == torch.Size([node_attr.size(0), self.out_channels])
        messages = node_mask.mm(edge_out)

        output = messages + root_info
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.node_in_channels) + ' -> ' \
               + str(self.out_channels) + ')'

class SparseNodeConv(Module):
    r"""
    Basic additive kernel GCN layer without edge features with sparse implementation.
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(SparseNodeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.node_weight = Parameter(torch.zeros([in_channels, out_channels], dtype=torch.float32))
        self.root = Parameter(torch.zeros([in_channels, out_channels], dtype=torch.float32))
        if bias:
            self.bias = Parameter(torch.zeros(out_channels, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        node_stdv = 1. / math.sqrt(self.node_weight.size(1))
        self.node_weight.data.uniform_(-node_stdv, node_stdv)
        self.root.data.uniform_(-node_stdv, node_stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-node_stdv, node_stdv)

    def forward(self, node_attr, node_mask):

        potential_node_msgs = torch.mm(node_attr, self.node_weight)
        sum_filter_node_msgs = torch.sparse.mm(node_mask, potential_node_msgs)
        root_info = torch.mm(node_attr, self.root)
        output = sum_filter_node_msgs + root_info
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'

class DuvenaudMP(MessagePassing):
    r"""
        message passing layer from  <https://proceedings.neurips.cc/paper/2015/file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf>
    """
    def __init__(self, in_channels, out_channels, dropout=0, bias=True):
        super(DuvenaudMP, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Linear(in_channels, out_channels)
        self.nonlin = ReLU()
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, edge_index):
        x = self.nonlin(self.weight(aggr_out))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x


def global_sum_pool(x, batch_mat):
    return torch.mm(batch_mat, x)


