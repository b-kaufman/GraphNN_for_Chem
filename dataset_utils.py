import os
import shutil
from random import uniform
import torch
import numpy as np



def split_data_dir(data_dir, train=0.7, valid=0.2, test=0.1, ext='.pt'):

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    dirs = [train_dir, valid_dir, test_dir]

    for direc in dirs:
        if not os.path.exists(direc):
            os.makedirs(direc)
        else:
            for tensor in os.listdir(direc):
                if ext in tensor:
                    old_path = os.path.join(direc, tensor)
                    new_path = os.path.join(data_dir, tensor)
                    shutil.move(old_path, new_path)

    for filename in os.listdir(data_dir):

        if ext in filename:
            old_path = os.path.join(data_dir, filename)
            val = uniform(0, 1)
            if val <= train:
                new_path = os.path.join(train_dir, filename)
            elif val <= train+valid:
                new_path = os.path.join(valid_dir, filename)
            else:
                new_path = os.path.join(test_dir, filename)
            shutil.move(old_path, new_path)


def adj_collate_help(batch, key, tot_nodes, num_node_features, num_edge_features=None):
    """

    :param batch: a list of tensors
    :param key: key associated with tensors in batch
    :return: batched tensors
    """

    accum = 0

    if key is "adj_mat":
        batch_adj_mat = torch.zeros((tot_nodes, tot_nodes))
        for datum in batch:
            adj_mat = datum['adj_mat']
            num_nodes = adj_mat.size()[0]
            batch_adj_mat[accum:accum + num_nodes, accum:accum + num_nodes] = adj_mat
            accum += num_nodes
        return batch_adj_mat

    if key is "atom_feat_mat" or "edge_feat_list_mat":
        return torch.stack(batch, 0)

    if key is "edge_feat_adj_mat":
        batch_feat_mat = torch.zeros((num_edge_features, tot_nodes, tot_nodes))
        for datum in batch:
            feat_mat = datum['edge_feat_adj_mat']
            num_nodes = feat_mat.size()[1]
            batch_feat_mat[:, accum:accum + num_nodes, accum:accum + num_nodes] = feat_mat
            accum += num_nodes
        return batch_feat_mat





def get_batch_tensor_and_edge_list(batch, e_list):
    accum = 0
    graph_sizes = np.zeros(len(batch)).astype(np.int64)
    edge_counts = np.zeros(len(batch)).astype(np.int64)
    for idx, datum in enumerate(batch):
        num_nodes = datum['atom_feat_mat'].size()[0]
        num_edges = datum['edge_feat_list_mat'].size()[0]
        graph_sizes[idx] = num_nodes
        edge_counts[idx] = num_edges
        accum += num_nodes
    batch_tens = torch.zeros([len(batch), accum])
    node_accum = 0
    edge_accum = 0
    for idx in range(graph_sizes.shape[0]):
        graph_nodes = graph_sizes[idx]
        graph_edges = edge_counts[idx]
        batch_tens[idx, node_accum:node_accum + graph_nodes] = 1
        e_list[:, edge_accum:edge_accum + graph_edges] += edge_accum
        node_accum += graph_nodes
        edge_accum += graph_edges

    return batch_tens, e_list

def get_batch_tensor(graph_list,total_nodes):
    batch_tens = torch.zeros([len(graph_list), total_nodes])
    accum = 0
    for idx, datum in enumerate(graph_list):
        num_graph_nodes = datum.size()[0]
        batch_tens[idx, accum:accum+num_graph_nodes] = 1
        accum += num_graph_nodes
    return batch_tens

def get_edge_list(edge_list_batch,total_edges):
    edge_batch = torch.zeros([2, total_edges],dtype=torch.int64)
    accum = 0
    max_node = 0
    for idx, datum in enumerate(edge_list_batch):
        num_edges = datum.size()[1]
        edge_batch[:, accum:accum+num_edges] = datum + edge_batch[:, accum:accum+num_edges] + max_node
        accum += num_edges
        max_node = torch.max(datum)
    return edge_batch

def edge_list_graph_collate(batch):

    graph_dict = {}

    atom_feat_mats = [d['atom_feat_mat'] for d in batch]
    graph_dict['num_graphs'] = len(atom_feat_mats)
    edge_feat_mats = [d['edge_feat_list_mat'] for d in batch]
    edge_lists = [d['edge_list'] for d in batch]

    graph_dict['atom_feat_mat'] = torch.cat(atom_feat_mats, 0)
    graph_dict['edge_feat_list_mat'] = torch.cat(edge_feat_mats, 0)
    graph_dict['label'] = torch.cat([d['label'] for d in batch], 0)

    tot_nodes = graph_dict['atom_feat_mat'].size()[0]
    tot_edges = graph_dict['edge_feat_list_mat'].size()[0]

    graph_dict['node_membership'] = get_batch_tensor(atom_feat_mats, tot_nodes)
    graph_dict['edge_list'] = get_edge_list(edge_lists, tot_edges)

    return graph_dict

def adj_graph_collate(batch):
    """
    likely belongs in own collate file
    :param batch:
    :return:
    """
    batch_tensor = get_batch_tensor(batch)
    total_nodes = batch_tensor.size()[1]
    node_features = batch[0]['atom_feat_mat'].size()[1]
    edge_features = None
    if 'edge_feat_adj_mat' in batch[0]:
        edge_features = batch[0]['edge_feat_adj_mat'].size()[0]
    graph_dict = {key: adj_collate_help([d[key] for d in batch], key, total_nodes, node_features, edge_features) for key
                  in
                  batch[0]}
    graph_dict['node_membership'] = batch_tensor
    return graph_dict