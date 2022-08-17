from abc import ABC, abstractmethod
import pandas as pd
from torch_geometric.data import Data
import moldata
import torch
import numpy as np
import os
import modifymol
from openeye import oechem

class MolTensor2D(object):

    def __init__(self, name, mol_rep, label, atom_feat_fun, bond_feat_func, vec_size=None):
        self.name = name
        self.mol_rep = mol_rep
        self.label = label
        self.adj_mat = moldata.generate_adjacency_mat(mol_rep)
        self.edge_list = moldata.generate_edge_list(mol_rep)
        self.atom_feat_mat = moldata.generate_atom_feature_matrix(self.mol_rep, atom_feat_fun, vec_size=vec_size)
        self.edge_feat_adj_mat = moldata.generate_bond_adj_feat_mat(self.mol_rep, bond_feat_func, vec_size=vec_size)
        self.edge_feat_list_mat = moldata.build_bond_tensor(self.mol_rep, bond_feat_func, vec_size=vec_size)
        self.graph_representation_dict = {'adj_mat': self.adj_mat, 'edge_list': self.edge_list,
                                          'node_attr': self.atom_feat_mat,
                                          'edge_feat_adj_mat': self.edge_feat_adj_mat,
                                          'edge_attr': self.edge_feat_list_mat,
                                          'y': self.label}

    def change_item_type(self,key,ktype):
        if ktype is 'float':
            self.graph_representation_dict[key] = self.graph_representation_dict[key].astype(float)

    def get_dict_torch_representation(self):
        dict_fun = lambda y: torch.from_numpy(self.graph_representation_dict[y]) \
            if not isinstance(self.graph_representation_dict[y], dict) \
            else {z: torch.from_numpy(self.graph_representation_dict[y][z]) for z in self.graph_representation_dict[y]}

        return {x: dict_fun(x) for x in self.graph_representation_dict}

    def writePyTorchTensor(self, data_dir):
        ext = '.pt'
        torch_dict = self.get_dict_torch_representation()
        torch.save(torch_dict, os.path.join(data_dir, self.name + ext))

    def writePyTorchGeomData(self, data_dir):
        ext = '.pt'
        torch_dict = self.get_dict_torch_representation()
        data = Data(x=torch_dict['atom_feat_mat'], edge_attr=torch_dict['edge_feat_list_mat'],
                    edge_index=torch_dict['edge_list'],
                    y=torch_dict['label'])
        torch.save(data, os.path.join(data_dir, self.name + ext))

    def writeH5PyTensor(self):
        pass

    def isvalid(self):
        for key in self.graph_representation_dict:
            if self.graph_representation_dict[key] is None:
                return False
        return True

def create_and_write_mol_graph_tensors(smiles_file, tensor_dir, label_col,label_type='binary',smiles_col='smiles',tensor_type='torch',delim='\t'):
    '''

    :param smiles_file: path to csv file containing smiles
    :param tensor_dir: absolute path of directory to write tensors to. Need not exist.
    :param label: pytorch tensor label associated with all tensors
    :param tensor_type: torch or torch_geo, based on base format.
    :return:
    '''

    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    df = moldata.load_smiles_df(smiles_file,delim=delim)
    #label = np.asarray([1., 0.]).reshape((1, 2)).astype(np.float32)
    for idx, row in df.iterrows():
        if label_type == 'binary':
            label = np.zeros((1,2)).astype(np.float32)
            ii = int(row[label_col])
            label[0,ii] = 1.
        if label_type == 'regression':
            label = np.zeros((1,1)).astype(np.float32)
            label[0] = row[label_col]
        #label = np.asarray((row[label_col])).astype(np.float32)
        tens = MolTensor2D('mol' + str(idx), row[smiles_col], label, moldata.get_basic_atom_vec, moldata.get_bond_vec,
                           vec_size=None)

        if tens.isvalid():
            tens.change_item_type('adj_mat', 'float')
            if tensor_type is 'torch':
                tens.writePyTorchTensor(tensor_dir)
            elif tensor_type is "torch_geo":
                tens.writePyTorchGeomData(tensor_dir)

def create_and_write_valid_invalid_mol_graph_tensors(smiles_file, tensor_dir, tensor_type='torch', max_class_size=None):

    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    df = moldata.load_smiles_df(smiles_file)
    label = np.asarray([1., 0.]).reshape((1, 2)).astype(np.float32)
    for idx, val in enumerate(df.smiles):
        tens = MolTensor2D('true' + str(idx), val, label, moldata.get_basic_atom_vec, moldata.get_bond_vec,
                           vec_size=None)

        if tens.isvalid():
            tens.change_item_type('adj_mat', 'float')
            if tensor_type is 'torch':
                tens.writePyTorchTensor(tensor_dir)
            elif tensor_type is "torch_geo":
                tens.writePyTorchGeomData(tensor_dir)
        if max_class_size and max_class_size == idx:
            break

    label = np.asarray([0., 1.]).reshape((1, 2)).astype(np.float32)
    for idx, val in enumerate(df.smiles):
        new_mol = modifymol.mutate_mol(val)
        if new_mol is None:
            continue
        tens = MolTensor2D('invalid' + str(idx), new_mol, label, moldata.get_basic_atom_vec, moldata.get_bond_vec,
                           vec_size=None)

        if tens.isvalid():
            tens.change_item_type('adj_mat', 'float')
            if tensor_type is 'torch':
                tens.writePyTorchTensor(tensor_dir)
            elif tensor_type is "torch_geo":
                tens.writePyTorchGeomData(tensor_dir)
        if max_class_size and max_class_size == idx:
            break

def get_label_dict(df, mol_tag, label_list):
    mol_info = df.loc[df['molecule_name'] == mol_tag]
    return {label_name: mol_info[label_name].values[0].astype(np.float32).reshape([1, 1]) for label_name in label_list}

def create_and_write_from_mol2(mol2_dir, data_dir, info_df, label_list, tensor_type='torch', max_class_size=None):

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    data = pd.read_pickle(info_df)
    labeled_molecules = data['molecule_name'].values
    for idx, file in enumerate(os.listdir(mol2_dir)):
        mol_tag = file.split('.')[0]
        if mol_tag not in labeled_molecules:
            continue
        label = get_label_dict(data, mol_tag, label_list)
        rd_mol = moldata.get_mol_from_mol2(os.path.join(mol2_dir, file),removeH=False)
        if rd_mol is None:
            continue
        tens = MolTensor2D(mol_tag, rd_mol, label, moldata.get_pos_atom_vec, moldata.get_bond_vec,
                           vec_size=None)

        if tens.isvalid():
            tens.change_item_type('adj_mat', 'float')
            if tensor_type is 'torch':
                tens.writePyTorchTensor(data_dir)
            elif tensor_type is "torch_geo":
                tens.writePyTorchGeomData(data_dir)
        if max_class_size and max_class_size == idx:
            break

    return